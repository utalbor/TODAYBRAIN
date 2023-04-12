"""Description: Lecture des données et extraction des features pour tous les canaux"""

import os
import pickle
import gudhi
import mne
import numpy as np
from Distances import D1, D2, PLV, PLI, MI, corr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# SIMILARITES = [corr, PLV, PLI, MI]  # Ensemble des fonctions de Similarité/Connectivités
# DISTANCES = [D1, D2]  # Ensemble des fonctions de Distance

'''Def path'''
data_path = '/Users/user/Documents/Projet/Dataset/'
diff = ['MATBeasy', 'MATBmed', 'MATBdiff']
dir_path = '/Users/user/Documents/Projet/MI/'
if not os.path.exists(dir_path):
    os.mkdir(dir_path)


def save_variable(v, filename):
    """
    Sauvegarder une variable comme un fichier(pour éviter de tourner les codes plusieurs fois)
    :param v: variable à sauvegarder
    :param filename: nom du fichier
    :return: le nom du fichier
    """
    f = open(dir_path + filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variable(filename):
    """
    Lire un fichier comme une variable
    :param filename: le nom du fichier qu'on veux lire
    :return: la variable
    """
    f = open(dir_path + filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


def read_epoch(personne, session, level, epoch):
    """
    Lecture d'une époch de données bruts
    :param personne: numéro de la personne(entre 1 et 15)
    :param session: la session qu'on veux lire(entre 1 et 2)
    :param level: niveau de charge de l'époch, appartient à diff[0/1/2]
    :param epoch: numéro de l'époch, entre 0 et 148
    :return: données d'une époch de taille (channels, time)
    """
    epochs_data = []
    path = os.path.join(data_path,
                        f'P{str(personne).zfill(2)}') + f'/S{session}/eeg/alldata_sbj{str(personne).zfill(2)}_sess{session}_{level}.set'
    # Read the epoched data with MNE
    epochs = mne.io.read_epochs_eeglab(path, verbose=False)
    tmp = epochs.get_data()
    epochs_data.extend(tmp)
    epochs_data = np.array(epochs_data)
    epochs_data = epochs_data[epoch]
    return epochs_data


def read_epoch149(personne, session, level):
    """
    Lecture de toutes les épochs d'un niveau de charge mentale
    :param personne: numéro de la personne(entre 1 et 15)
    :param session: la session qu'on veux lire(entre 1 et 2)
    :param level: niveaux de charge, appartient à diff[0/1/2]
    :return: données d'un niveau de charge mentale de taille (epochs, channels, time)
    """
    epochs_data = []
    path = os.path.join(data_path,
                        f'P{str(personne).zfill(2)}') + f'/S{session}/eeg/alldata_sbj{str(personne).zfill(2)}_sess{session}_{level}.set'
    # Read the epoched data with MNE
    epochs = mne.io.read_epochs_eeglab(path, verbose=False)
    tmp = epochs.get_data()
    epochs_data.extend(tmp)
    epochs_data = np.array(epochs_data)
    return epochs_data


def bettiCurve(D, max_dim, nb_point):
    """
    Trace la courbe de betti number à partir de la persistence
    :param D: la persistence calculée
    :param max_dim: dimension maximale choisi (souvent 3 -> betti0, betti1 et betti2)
    :param nb_point: nb des points voulus
    :return: la courbe de betti number
    """
    max = 0
    for i in range(len(D)):
        if D[i][1][0] > max:
            max = D[i][1][0]

    Echelle = np.linspace(0, max, nb_point)
    Betti = np.zeros((max_dim, nb_point))
    for i in range(max_dim):
        for j in range(len(D)):
            if D[j][0] == i:
                for k in range(len(Echelle)):
                    if D[j][1][0] <= Echelle[k] < D[j][1][1]:
                        Betti[i][k] += 1
    return Betti

def Betti_Curve(Betti,max_dim):
    """Trace les courbes de bettis numbers à partir des Bettis numbers obtenu par la fonction bettiCurve

    Args:
        Betti (matrix): matrice de dimension (max_dim,nb_point) contenant les bettis numbers
        max (int): maximum de dimension
    """
    nb_point = np.shape(Betti)
    Echelle = np.linspace(0,max,nb_point)
    plt.figure()
    color = ['r','b','g']
    for i in range(max_dim):
        plt.plot(Echelle,Betti[i],color[i])
    B = ['Betti'+str(i) for i in range(0,max_dim+1)]
    plt.legend(B)
    plt.ylabel('Betti number')
    plt.xlabel('Density')
    plt.title('Betti Curve')
    
    
def logistic_function(x, A, L, k, x0):
    """Définition d'une fonction logisitique paramétrée afin de faire une interpolation avec la courbe de betti 0

    Args:
        x (int): variable
        A (int): Amplitude
        L (int): Hauteur
        k (int): coefficient de pente
        x0 (int): position du point d'inflexion

    Returns:
        _int_: valeur de la fonction logistique
    """
    return A*(1 - L / (1 + np.exp(-k * (x - x0))))

def traitement_trainingdata(nomdufichier):
    """
    Extraction des features (max et argmax) d'un fichier(Etant donnée un fichier qui sauvegarde la courbe de betti,
    on peut utiliser cette fonction pour retirer les features)
    :param nomdufichier: le nom du fichier
    :return: les max et argmax des courbes de betti 1 et 2.
    """
    Betti = load_variable(nomdufichier)
    Betti = np.array(Betti)
    Betti_max = np.zeros((149, 2, 1))
    Betti_argmax = np.zeros((149, 2, 1))
    for bettinum in range(2):
        for i in range(149):
            Betti_max[i, bettinum, 0] = np.max(Betti[i, bettinum + 1, :])  # Betti1
            Betti_argmax[i, bettinum, 0] = np.argmax(Betti[i, bettinum + 1, :])
    #for i in range(149): ##Rajoute une feature qui est la pente de la courbe de betti 0 
    ###  /!\ le code n'est pas adaptée si utilisé dans d'autres fonctions.
    #    Y = Betti[i,0,:]
    #    X = range(len(Y))
    #    popt, pcov= curve_fit(logistic_function,X,Y)
    #    [A,L,k,x0] = popt
    #    Betti_pente[i,0] = k     

    return Betti_max, Betti_argmax, #Betti_pente


def concatenate_data(n_person, classes, type):
    """
    Concatene les features retirées
    :param n_person: numéro de la personne
    :param classes: combien de niveaux de charges
    :param type: quel type de distance
    :return:
    """
    if classes == 3:
        label = np.concatenate((np.zeros(149), np.ones(149), 2 * np.ones(149)))
        levels = [0, 1, 2]
    elif classes == 2:
        label = np.concatenate((np.zeros(149), np.ones(149)))
        levels = [0, 1]  # 这里本来是【0，2】，如果先全部读取三个类别后去高低时要用0和2
    else:
        print('classes must be 2 or 3')
        levels = []
        label = 0

    data = []
    for level in levels:
        Betti_max, Betti_argmax = traitement_trainingdata(
            f'Person{n_person}_{type}_{level}.txt')  # rules for name a file
        Betti_features = np.concatenate((Betti_max, Betti_argmax), axis=1)
        data.append(Betti_features)

    data = np.array(data)
    data = data.reshape((classes * 149, -1))
    return data, label


if __name__ == "__main__":
    '''Définir les params'''
    max_dim = 3
    nb_point = 101
    Betti_mean = np.zeros((3, 3, 101))
    Betti_std = np.zeros((3, 3, 101))
    SIMILARITE = MI  #SIMILARITE CHOISIE
    DISTANCE = D2 #DISTANCE CHOISIE
    type = 'MI'

    '''Lire les features et sauvegarder'''
    for i in range(15):
        epochs_data = [read_epoch149(i + 1, 2, diff[0]), read_epoch149(i + 1, 2, diff[2])]
        for level in [0, 1]:  # if is two-classes
            color = ['red', 'blue', 'orange', 'black']
            Bettis = []
            for j in range(149):
                # eeg = eeg_f[i]: same
                eeg = epochs_data[level][j]
                D = DISTANCE(SIMILARITE(eeg))
                rips_complex = gudhi.RipsComplex(distance_matrix=D, max_edge_length=1)
                simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
                diag_D = simplex_tree.persistence()
                Betti = bettiCurve(diag_D, max_dim, nb_point)
                Bettis.append(Betti)
            save_variable(Bettis, f'Person{i + 1}_{type}_{level}.txt')
            # Betti_mean[level] = np.mean(Bettis, axis=0)
            # Betti_std[level] = np.std(Bettis, axis=0)
            Echelle = [j / (nb_point - 1) for j in range(nb_point)]

        X, y = concatenate_data(i + 1, 2, type)
        print(X.shape, y.shape)
        np.save(dir_path + f'Person{i + 1}_X.npy', X)
        np.save(dir_path + f'Person{i + 1}_y.npy', y)
