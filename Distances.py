"""Description: Dans ce fichier, on définit les distances qu'on va utiliser"""
"Les fonctions de similarités sont [corr, PLV, PLI, MI] et les distances sont [D1, D2]"
import sys
from scipy.signal import hilbert
import numpy as np

## Matrices de Similarités
def corr(eeg):
    S = np.corrcoef(eeg)
    return S

def phi(eeg):
    m, n = np.shape(eeg)
    PHI = np.zeros([n, m, m])
    hilbert_eeg = eeg + 1j * hilbert(eeg)
    for i in range(m):
        for j in range(m):
            hilbertsig1 = eeg[i] + 1j * hilbert(eeg[i])
            hilbertsig2 = eeg[j] + 1j * hilbert(eeg[j])
            for t in range(n):
                deltaphitij = np.angle(
                    (hilbertsig1[t] * np.conj(hilbertsig2[t])) / (np.abs(hilbertsig1[t]) * np.abs(hilbertsig2[t])))
                PHI[t, i, j] = deltaphitij
    return PHI

def PLV(eeg):
    PLVt = np.exp(1j * phi(eeg))
    PLV = 1 / len(eeg[0]) * np.abs(np.sum(PLVt, axis=0))
    return PLV

def PLI(eeg):
    PLIt = np.sign(phi(eeg))
    PLI = 1 / len(eeg[0]) * np.abs(np.sum(PLIt, axis=0))
    return PLI


def entropy(bins, *X):
    # binning of the data
    data, *edges = np.histogramdd(X, bins=bins)

    # calculate probabilities
    data = data.astype(float) / data.sum()

    # compute H(X,Y,...,Z) = sum(-P(x,y,...,z) ∗ log2(P(x,y,...,z)))
    return np.sum(-data * np.log2(data + sys.float_info.epsilon))


def mutual_information(bins, X, Y):
    # compute I(X,Y) = H(X) + H(Y) − H(X,Y)

    H_X = entropy(bins, X)
    H_Y = entropy(bins, Y)
    H_XY = entropy(bins, X, Y)

    return H_X + H_Y - H_XY


# Compute number of bins using Sturge's rule
def compute_mi_matrix(eeg):
    """ Compute Mutual Information matrix.

        Return: mi_matrix
    """
    n_cols = eeg.shape[0]
    mi_matrix = np.zeros([n_cols, n_cols])

    # Sturge's rule for number of bins
    n_bins = int(1 + 3.322 * np.log10(eeg.shape[1]))

    for i in range(n_cols):
        for j in range(n_cols):
            mi = mutual_information(n_bins, eeg[i, :], eeg[j, :])
            mi_matrix[i, j] = mi

    return mi_matrix


def compute_normed_mi_matrix(mi_matrix):
    divisor_matrix = np.sqrt(np.diag(mi_matrix) * np.diag(mi_matrix).reshape(-1, 1))
    normed_mi_matrix = mi_matrix / divisor_matrix
    return normed_mi_matrix


def MI(eeg):
    return compute_normed_mi_matrix(compute_mi_matrix(eeg))

## Matrices de Distances

def D1(S):
    D = np.sqrt(-np.log(S))
    return D

def D2(S):
    D = np.ones(np.shape(S)) - np.abs(S)

