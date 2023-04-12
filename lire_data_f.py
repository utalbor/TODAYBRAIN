"""Description: Lecture des données et extraction des features pour les canaux frontaux uniquement"""
"A vérifier qu'il s'agit bien des canaux frontaux"
import os
import pickle
import gudhi
import mne
import numpy as np
from Distances import D1, D2, PLV, PLI, MI

# SIMILARITES = [corr, PLV, PLI, MI]  # Ensemble des fonctions de Similarité/Connectivités
# DISTANCES = [D1, D2]  # Ensemble des fonctions de Distance

'''Def path'''
data_path = '/Users/user/Documents/Projet/Dataset/'
diff = ['MATBeasy', 'MATBmed', 'MATBdiff']
dir_path = '/Users/user/Documents/Projet/MI-front/'
if not os.path.exists(dir_path):
    os.mkdir(dir_path)


def save_variable(v, filename):
    """
    Sauvegarde d'une variable comme un fichier(pour éviter de tourner les codes plusieurs fois)
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
    Leture d'un fichier comme une variable
    :param filename: le nom du fichier qu'on veux lire
    :return: la variable
    """
    f = open(dir_path + filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


def read_epoch(personne, session, level, epoch):
    """
    Lire une époch de données brutes
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


def traitement_trainingdata(nomdufichier):
    """
    Extraction des features (max et argmax) d'un fichier(Etant donnée un fichier qui sauvegarde la courbe de betti,
    on peut utiliser cette fonction pour retirer les features)
    :param nomdufichier: le nom du fichier
    :return: les max et argmax des courbes de betti 1 et 2
    """
    Betti = np.load(nomdufichier)
    Betti = np.array(Betti)
    Betti_max = np.zeros((149, 2, 1))
    Betti_argmax = np.zeros((149, 2, 1))
    for bettinum in range(2):
        for i in range(149):
            Betti_max[i, bettinum, 0] = np.max(Betti[i, bettinum + 1, :])  # Betti1
            Betti_argmax[i, bettinum, 0] = np.argmax(Betti[i, bettinum + 1, :])

    return Betti_max, Betti_argmax


def concatenate_data(n_person, classes, type):
    """
    Concatenate les features retirées
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
        levels = [0, 1]
    else:
        print('classes must be 2 or 3')
        levels = []
        label = 0

    data = []
    for level in levels:
        Betti_max, Betti_argmax = traitement_trainingdata(dir_path +
                                                          f'Person{n_person}_{type}_{level}.npy')  # rules for name a file
        Betti_features = np.concatenate((Betti_max, Betti_argmax), axis=1)
        data.append(Betti_features)

    data = np.array(data)
    data = data.reshape((classes * 149, -1))
    return data, label


# Dont know where is the electrode which could not be used
# ECG1 is between 7 and 27
# reference is the 64
# On suppose 1～7 26~38 56~61 sont fontales, unused between 38-56)

def extract_rows(matrix, row_indices):
    """
    Extraction des colonnes envisagées
    :param matrix: matrice à extraire
    :param row_indices: les colonnes envisagées
    :return: matrice après l'extraction
    """
    return np.take(matrix, row_indices, axis=1)


if __name__=="__main__":
    '''Définir les params'''
    max_dim = 3
    nb_point = 101
    Betti_mean = np.zeros((3, 3, 101))
    Betti_std = np.zeros((3, 3, 101))
    SIMILARITE = MI
    DISTANCE = D2
    type = 'MI-f'
    index = np.concatenate((np.arange(1, 7), np.arange(26, 38), np.arange(56, 61)))

    '''Lire les features et sauvegarder'''
    for i in range(15):
        epochs_data = [read_epoch149(i + 1, 2, diff[0]), read_epoch149(i + 1, 2, diff[2])]
        for level in [0, 1]:  # if is two-classes
            color = ['red', 'blue', 'orange', 'black']
            Bettis = []
            eeg_f = extract_rows(epochs_data[level], index)
            for j in range(149):
                # eeg = eeg_f[i]: same
                eeg = eeg_f[j]
                print(eeg.shape)
                D = DISTANCE(SIMILARITE(eeg))
                rips_complex = gudhi.RipsComplex(distance_matrix=D, max_edge_length=1)
                simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
                diag_D = simplex_tree.persistence()
                Betti = bettiCurve(diag_D, max_dim, nb_point)
                Bettis.append(Betti)
            np.save(dir_path + f'Person{i + 1}_{type}_{level}.npy', Bettis)
            # Betti_mean[level] = np.mean(Bettis, axis=0)
            # Betti_std[level] = np.std(Bettis, axis=0)
            Echelle = [j / (nb_point - 1) for j in range(nb_point)]

        X, y = concatenate_data(i + 1, 2, type)  # combine les features
        print(X.shape, y.shape)
        np.save(dir_path + f'Person{i + 1}_X.npy', X)
        np.save(dir_path + f'Person{i + 1}_y.npy', y)
