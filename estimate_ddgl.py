import time
import copy
import numpy as np
import matlab.engine
import scipy.io

eng = matlab.engine.start_matlab()
eng.cd(r'/home/user/Desktop/BrainMAGIC/Graph_codes/', nargout=0)


#  Function for finding the best generalized Laplacian estimate under connectivity constraints
#  This function returns the optimal solution of the GGL problem in [1].
#     Reference:
#     [1] H. E. Egilmez, E. Pavez, and A. Ortega, ?Graph learning from data under structural and Laplacian constraints,?
#        CoRR, vol. abs/1611.05181v1,2016. [Online]. Available: https://arxiv.org/abs/1611.05181v1
#
#  Inputs: S - sample covariance/input matrix
#          A_mask - 0/1 adjacency matrix where 1s represent connectivity
#          alpha -  regularization parameter
#          prob_tol - error tolerance for the optimization problem (stopping criterion)
#          inner_tol - error tolerance for the nonnegative QP solver
#          max_cycle - max. number of cycles
#          regularization_type - two options:
#                                1 (default): | O |_1 (l_1 regularization)
#                                2          : | O |_1,off (l_1 regularization on off-diagonal entries)
#
#  Outputs:
#  O   -   estimated generalized graph Laplacian (Theta)
#  C   -   estimated inverse of O
#  convergence  - is the structure with fields 'frobNorm', 'time':
#                 'frobNorm': vector of normalized differences between esimtates of O between cycles: |O(t-1)-O(t)|_F/|O(t-1)|_F
#                 'time':  runtime of each cycle
#
#
#   (C) Hilmi Enes Egilmez

def estimate_ddgl(S, A_mask, alpha=0, prob_tol=1e-4, inner_tol=1e-6, max_cycle=20, regularization_type=1):
    # variables
    n = np.shape(S)[1]
    if regularization_type == 1:
        H_alpha = (alpha) * (2 * np.eye(n) - np.ones(n))
    elif regularization_type == 2:
        H_alpha = (alpha) * (np.eye(n) - np.ones(n))
    else:
        print("regularization_type can be either 1 or 2.")
        return;
    K = S + H_alpha
    K = K.astype(float)

    # starting value;
    O_init = np.diag((1 / np.diag(K)))  # S \ eye(p);
    C = np.diag(np.diag(K))
    C = C.astype(float)
    O = copy.deepcopy(O_init)
    O = O.astype(float)
    # Best output
    O_best = copy.deepcopy(O)
    C_best = copy.deepcopy(C)

    sol = np.trace(np.matmul(O_init, K)) - np.linalg.slogdet(O_init)[1]
    print('minimisation:', sol)

    frob_norm = []
    time_counter = []
    converged = False
    cycle = 0

    while (not converged) and cycle < max_cycle:

        O_old = copy.deepcopy(O)
        t = time.time()
        # Inner loop
        for u in range(n):
            minus_u = np.setdiff1d(range(n), u)  # index of u complement

            # input matrix variables
            k_uu = K[u, u]
            k_u = K[minus_u, u]

            # update Ou_inv
            c_u = C[minus_u, u]
            c_u = c_u.astype(float)
            c_uu = C[u, u]
            Ou_i = C[np.ix_(minus_u, minus_u)] - (
                        np.matmul(np.expand_dims(c_u, axis=1), np.expand_dims(c_u, axis=1).T) / c_uu)

            # block-descent variables
            beta = np.zeros([n - 1, 1])
            ind_nz = A_mask[minus_u, u] == 1  # non-zero indices
            A_nnls = Ou_i[np.ix_(ind_nz, ind_nz)]
            b_nnls = k_u[ind_nz] / k_uu

            # block-descent step

            out = eng.nonnegative_qp_solver(matlab.double(A_nnls.tolist()),
                                            matlab.double(np.expand_dims(b_nnls, axis=1).tolist()), inner_tol);
            beta_nnls = -np.asarray(out['xopt'])  # sign flip
            beta[ind_nz] = beta_nnls
            o_u = copy.deepcopy(beta).astype(float)
            o_uu = (1 / k_uu) + np.matmul(o_u.T, np.matmul(Ou_i, o_u))
            # Update the current Theta
            O[u, u] = o_uu
            O[np.expand_dims(minus_u, axis=1), u] = o_u
            O[u, np.expand_dims(minus_u, axis=1)] = o_u
            # print (O)

            # Update the current Theta inverse
            cu = np.matmul(Ou_i, o_u) / (o_uu - np.matmul(o_u.T, np.matmul(Ou_i, o_u)))
            cuu = 1 / (o_uu - np.matmul(o_u.T, np.matmul(Ou_i, o_u)))
            C[u, u] = cuu  # C[u,u] = k_uu

            C[u, np.expand_dims(minus_u, axis=1)] = -cu
            C[np.expand_dims(minus_u, axis=1), u] = -cu

            # use Sherman-Woodbury
            C[np.ix_(minus_u, minus_u)] = (Ou_i + (np.matmul(cu, cu.T) / cuu))

        if cycle > 3:
            d_shifts = np.squeeze(np.matmul(O, np.ones((n, 1))))
            neg_diag_idx = np.where(d_shifts < 0)[0]
            for idx_t in range(np.shape(neg_diag_idx)[0]):
                idx = neg_diag_idx[idx_t]
                [O, C] = update_sherman_morrison_diag(O, C, -d_shifts[idx], idx)

        O_best = copy.deepcopy(O)
        C_best = copy.deepcopy(C)
        # Equation de minimisation
        # sol= np.trace(np.matmul(O,K))-np.linalg.slogdet(O)[1]
        cycle = cycle + 1
        # print('minimisation:',sol)

        # %%% time
        time_counter.append(time.time() - t)
        # %%% calculate frob norms
        frob_norm = np.append(frob_norm, np.linalg.norm(O_old - O, 'fro') / np.linalg.norm(O_old, 'fro'))
        print('Cycle: {}     |Theta(t-1)-Theta(t)|_F/|Theta(t-1)|_F = {}'.format(cycle, (frob_norm[-1])))

        if cycle > 5:
            # % convergence criterions (based on theta)
            if (((frob_norm[-1])) < prob_tol):
                converged = True
                O_best = copy.deepcopy(O)
                C_best = copy.deepcopy(C)
    O = copy.deepcopy(O_best)
    C = copy.deepcopy(C_best)
    return O, C


#  Sherman Morrison
#  Inputs:   O = current target, C= current inverse
#            shift, idx: amount of shift, index
#  Outputs:  O = new target, C= new inverse

def update_sherman_morrison_diag(O,C,shift,idx):
    O[idx,idx] = O[idx,idx] + shift
    c_d = C[idx,idx]
    C = C - np.matmul((np.expand_dims(C[:,idx], axis=1)*shift),(np.expand_dims(C[idx,:], axis=0)))/(1+shift*c_d)
    return O,C

