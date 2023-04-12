'''Description: Classification des deux niveaux de charge mentale en utilisant les algorithmes de ML classiques'''
import numpy as np
from Distances import D1, D2, corr, PLV, PLI, MI
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('qt5agg')

'''Définition des paramètres'''
SIMILARITES = [corr, PLV, PLI, MI]  # Ensemble des fonctions de Similarité/Connectivités
DISTANCES = [D1, D2]  # Ensemble des fonctions de Distance
dir_path = '/Users/user/Documents/Projet/MI/'
dir_path2 = '/Users/user/Documents/Projet/MI-front/'
df = pd.DataFrame()

'''
Classification des features en utilisant SVM, Random forêt et Régression logistique
Utilisation du 10-Fold cross validation
Sauvegarde des accuracy de chaque méthode de chaque sujets dans un tableau
'''
# pour tous les canaux
for i in range(15):
    # load les features
    X = np.load(dir_path+f'Person{i+1}_X.npy')
    y = np.load(dir_path+f'Person{i+1}_y.npy')

    # définir les classifieurs
    clf_svm = SVC()
    clf_rf = RandomForestClassifier()
    clf_lr = LogisticRegression()

    # 10 fold cross validation
    kf = KFold(n_splits=10, shuffle=True)
    scores_svm = cross_val_score(clf_svm, X, y, cv=kf)
    scores_rf = cross_val_score(clf_rf, X, y, cv=kf)
    scores_lr = cross_val_score(clf_lr, X, y, cv=kf)

    # sauvegarder dans un tableau
    temp_df = pd.DataFrame({'Model': ['SVM'] * 10, 'Accuracy': scores_svm, 'Sujet': [i+1] * 10})
    df = pd.concat([df, temp_df], ignore_index=True)
    temp_df = pd.DataFrame({'Model': ['RandomForest'] * 10, 'Accuracy': scores_rf, 'Sujet': [i+1] * 10})
    df = pd.concat([df, temp_df], ignore_index=True)
    temp_df = pd.DataFrame({'Model': ['LogisticRegression'] * 10, 'Accuracy': scores_lr, 'Sujet': [i + 1] * 10})
    df = pd.concat([df, temp_df], ignore_index=True)

# pour les channaux fontaux
for i in range(15):
    # load les features-frontales
    X = np.load(dir_path2+f'Person{i+1}_X.npy')
    y = np.load(dir_path2+f'Person{i+1}_y.npy')

    # définir les classifieurs
    clf_svm = SVC()
    clf_rf = RandomForestClassifier()
    clf_lr = LogisticRegression()

    # 10 fold cross validation
    kf = KFold(n_splits=10, shuffle=True)
    scores_svm = cross_val_score(clf_svm, X, y, cv=kf)
    scores_rf = cross_val_score(clf_rf, X, y, cv=kf)
    scores_lr = cross_val_score(clf_lr, X, y, cv=kf)

    # sauvegarder dans un tableau
    temp_df = pd.DataFrame({'Model': ['SVM-front'] * 10, 'Accuracy': scores_svm, 'Sujet': [i+1] * 10})
    df = pd.concat([df, temp_df], ignore_index=True)
    temp_df = pd.DataFrame({'Model': ['RandomForest-front'] * 10, 'Accuracy': scores_rf, 'Sujet': [i+1] * 10})
    df = pd.concat([df, temp_df], ignore_index=True)
    temp_df = pd.DataFrame({'Model': ['LogisticRegression-front'] * 10, 'Accuracy': scores_lr, 'Sujet': [i + 1] * 10})
    df = pd.concat([df, temp_df], ignore_index=True)


'''Retirer les outliers'''
print(df.head(30))
print(df.groupby('Model')['Accuracy'].mean())
df = df.sort_values(by='Accuracy', ascending=False)
grouped = df.groupby(['Model', 'Sujet']).head(7)
grouped = grouped.sort_values(by='Accuracy', ascending=True)
grouped2 = grouped.groupby(['Model', 'Sujet']).head(5)
print(grouped2.groupby('Model')['Accuracy'].mean())


'''Boxplot des accuracys'''
x = "Sujet"
y = "Accuracy"

# ax = sns.boxplot(data=grouped, x=x, y=y, hue='Model')
# plt.title('Comparaison tous modèles pour la distance MI')
# plt.show()
