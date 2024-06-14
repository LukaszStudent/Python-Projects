from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#zadanie 1a
arr = np.dot(np.random.rand(200,2), np.random.randn(2,2))

#zadanie 1b
plt.scatter(arr[:,0],arr[:,1])
plt.show()

#zadanie 1c
def wiPCA(zbior,rozmiar):
    wiersze,kolumny=zbior.shape
    standaryzowany_zbior=np.zeros(shape=(wiersze,kolumny))
    for kolumna in range(kolumny):
        srednia=np.mean(zbior[:,kolumna])
        odchylenie=np.std(zbior[:,kolumna])
        tempArray=np.empty(0)
        for element in zbior[:,kolumna]:
            tempArray = np.append(tempArray,((element-srednia)/odchylenie))
        standaryzowany_zbior[:,kolumna] = tempArray
    #print('Standaryzowany zbior: \n',standaryzowany_zbior,'\n')
    kowariancja=np.cov(standaryzowany_zbior.T)
    #print(kowariancja)
    #poroblem z zadaniem 3
    kowariancja=np.nan_to_num(kowariancja,nan=0.0)
    eigenvalues,eigenvect = LA.eig(kowariancja)
    P = eigenvect.T.dot(standaryzowany_zbior.T)
    #print('P',P)
    #print(eigenvalues)

    PC1=(max(eigenvalues)/sum(eigenvalues))
    PC2=1-PC1

    kolejne_wektory=[]
    for i in eigenvalues:
        kolejne_wektory.append((i/sum(eigenvalues))*100)
    #print(kolejne_wektory)
    kolejne_wymiary=np.cumsum(kolejne_wektory)


    plt.scatter(P.T[:,0],P.T[:,1])
    plt.show()
    sns.lineplot(x=np.linspace(1,len(eigenvalues),len(eigenvalues)),y=kolejne_wymiary)
    plt.xlabel('numery składowej')
    plt.ylabel('skumulowana wariancja [%]')
    plt.show()

wiPCA(arr,2)


#zadanie 2a
from sklearn import datasets
from sklearn.decomposition import PCA
iris=datasets.load_iris()
pca=PCA(n_components=2)
X_r=pca.fit_transform(iris.data)

wiPCA(iris.data,2)


#zadanie 3
from sklearn import datasets
from sklearn.decomposition import PCA
digits=datasets.load_digits()

wiPCA(digits.data,2)
