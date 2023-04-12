'''Description: Classifier les deux niveaux de charge mentale en utilisant les algorithmes de ML classiques'''
import numpy as np
from Distances import D1, D2,corr, PLV, PLI, MI
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('qt5agg')

'''Définitions des paramètres'''
SIMILARITES = [corr, PLV, PLI, MI]  # Ensemble des fonctions de Similarité/Connectivités
DISTANCES = [D1, D2]  # Ensemble des fonctions de Distance
dir_path = '/Users/user/Documents/Projet/MI/'
df = pd.DataFrame()

'''
Classification ses features en utilisant SVM, Random forêt et Régression logistique
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


'''Retirer les outliers'''
print(df.groupby('Model')['Accuracy'].mean())
df = df.sort_values(by='Accuracy', ascending=False)
grouped = df.groupby(['Model', 'Sujet']).head(7)
grouped = grouped.sort_values(by='Accuracy', ascending=True)
grouped2 = grouped.groupby(['Model', 'Sujet']).head(5)
print(grouped2.groupby('Model')['Accuracy'].mean())
# # df.drop(df[df['Sujet'] == 10].index, inplace=True)
# print(df.groupby(['Model', 'Sujet'])['Accuracy'].min())
# df.drop(df[df.groupby(['Model', 'Sujet'])['Accuracy'].min()].index, inplace=True)

'''Boxplot des accuracys'''
x = "Sujet"
y = "Accuracy"

# ax = sns.boxplot(data=grouped2, x=x, y=y, hue='Model')
# plt.title('Résultat de Distance de corrélation')
# plt.show()

