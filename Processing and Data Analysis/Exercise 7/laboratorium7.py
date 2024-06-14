#import numpy as np
#import scipy.stats
#import sklearn
#from sklearn import datasets
#import matplotlib.pyplot as plt


#class KNN_klasyfikacja():
#    def __init__(self, n_neighbors=1,use_KDTree=False):
#        self.k = n_neighbors
#        self.use_KDTree = use_KDTree

#    def fit(self,X,y):
#        self.inX=X
#        self.y=y
#        #self.inX=self.inX.values
#        #self.y=self.y.values

#    def predict(self,X):
#        dystans=np.linalg.norm(self.inX-X,axis=1)
#        nearest_neighbor=dystans.argsort()[:self.k]
#        nearest_neighbor_szukana=self.y[nearest_neighbor]
#        prediction=scipy.stats.mode(nearest_neighbor_szukana)
#        print(prediction)
#        return prediction

#    def score(self,X,y):
#        pass



#class KNN_regresja():
#    def __init__(self,n_neighbors=1,use_KDTree=False):
#        self.k = n_neighbors
#        self.use_KDTree = use_KDTree

#    def fit(self,X,y):
#        self.inX=X
#        self.y=y
#        #self.inX=self.inX.values
#        #self.y=self.y.values

#    def predict(self,X):
#        dystans=np.linalg.norm(self.inX-X,axis=1)
#        nearest_neighbor=dystans.argsort()[:self.k]
#        nearest_neighbor_szukana=self.y[nearest_neighbor]
#        prediction=nearest_neighbor_szukana.mean()
#        print(prediction)
#        print(type(prediction))
#        return prediction

#    def score(self,X,y):
#        suma=0
#        print(X)
#        print(y)

#        for i in range(0,len(X),1):
#          print(y[i])
#          print(X[i])
#          print('\n')
#          difference=X[i]-y[i]
#          squared_difference=difference**2
#          suma+=squared_difference
#          print(i)
#        MSE=suma/len(y)
#        print("The Mean Square Error is: " , MSE)
        
        
   

## 3.Klasyfikacja
#X, y = datasets.make_classification(n_samples=100,n_features=2,n_informative=2,n_redundant=0,n_repeated=0,random_state=3)
#print(X)
#print(y)

#klasyfikacja3=KNN_klasyfikacja(22)
#klasyfikacja3.fit(X,y)
#klasyfikacja3.predict(X)


#import pandas as pd

#url=('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data')

#abalone=pd.read_csv(url,header=None)
#print(abalone.head())
#abalone.columns=["Sex",
#    "Length",
#    "Diameter",
#    "Height",
#    "Whole weight",
#    "Shucked weight",
#    "Viscera weight",
#    "Shell weight",
#    "Rings",
#]
#abalone=abalone.drop('Sex',axis=1)
#print(abalone)
#import matplotlib.pyplot as plt
#plt.hist(abalone['Rings'],bins=15)
#plt.show()

#import numpy as np
#correlation_matrix=abalone.corr()
#print(correlation_matrix['Rings'])


#print('\ntabelka\n')
#print(abalone)
#X=abalone.drop('Rings',axis=1)

#y=abalone['Rings']


#new_data_point = np.array([ #dane nowego zwierzaka
#    0.569552,
#    0.446407,
#    0.154437,
#    1.016849,
#    0.439051,
#    0.222526,
#    0.291208,
#])


#print('\n\n\n\n\n KLASA\n')
#regresja=KNN_regresja(40)
#regresja.fit(X,y)
#regresja.predict(new_data_point)
#regresja.score(new_data_point,y)

#print('\n\n\n\n')
#klasyfikacja=KNN_klasyfikacja(100)
#klasyfikacja.fit(X,y)
#klasyfikacja.predict(new_data_point)


## 4.Regresja
#print('\nZADANIE 4\n')
#X,y=sklearn.datasets.make_regression()
#print(X)
#print(y)
#regresja_4=KNN_regresja(3)
#regresja_4.fit(X,y)
#regresja_4.predict(y)
#print(y.size)
#print(y.shape)

#print(X.size)
#print(X.shape)
#plt.scatter(X[:0],X[:1])
#plt.plot()


import matplotlib.pyplot as plt
#from matplotlib import style
#style.use('ggplot')
#import numpy as np

#X = np.array([[1, 2],
#              [1.5, 1.8],
#              [5, 8 ],
#              [8, 8],
#              [1, 0.6],
#              [9,11]])

#plt.scatter(X[:,0], X[:,1], s=150)
#plt.show()

#colors = 10*["g","r","c","b","k"]

import numpy as np
class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
print(iris)
import pandas as pd

df=pd.DataFrame(X)
print(df)

K=K_Means()
z=K.fit(X[:,1])




