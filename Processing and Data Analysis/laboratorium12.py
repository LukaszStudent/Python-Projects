from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.random import permutation

#2. Klasteryzacja
#2.1
iris=datasets.load_iris()
X=iris.data
y=iris.target
print('Y_real\n',y,'\n')

#2.2
clustering_ward=AgglomerativeClustering(linkage='ward').fit(X)
clustering_complete=AgglomerativeClustering(linkage='complete').fit(X)
clustering_average=AgglomerativeClustering(linkage='average').fit(X)
clustering_single=AgglomerativeClustering(linkage='single').fit(X)

#2.3
Y_pred_ward=clustering_ward.labels_
Y_pred_complete=clustering_complete.labels_
Y_pred_average=clustering_average.labels_
Y_pred_single=clustering_single.labels_

def find_perm(clusters, Y_real, Y_pred):
    perm=[]
    for i in range(clusters):
        idx = Y_pred == i
        new_label=stats.mode(Y_real[idx])[0][0]
        print(type(new_label))
        perm.append(new_label)
    return [perm[label] for label in Y_pred]

X_perm_ward=find_perm(2,y,Y_pred_ward)
X_perm_complete=find_perm(2,y,Y_pred_complete)
X_perm_average=find_perm(2,y,Y_pred_average)
X_perm_single=find_perm(2,y,Y_pred_single)


pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)
print(X_reduced.shape)

plt.subplot(1,3,1)
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y)

#yX=np.interp(X[:,1],(np.min(X[:,1]),np.max(X[:,1])),(X_reduced[:,1].min(),X_reduced[:,1].max()))
#xX=np.interp(X[:,0],(np.min(X[:,0]),np.max(X[:,0])),(X_reduced[:,0].min(),X_reduced[:,0].max()))
plt.subplot(1,3,2)
#plt.scatter(X[:,0]-6,X[:,1]-3,c=y)
kmeans2=KMeans(n_clusters=2)
X_reduced_KM2=kmeans2.fit_transform(X_reduced)
kmeans3=KMeans(n_clusters=3)
X_reduced_KM3=kmeans3.fit_transform(X_reduced)
kmeans4=KMeans(n_clusters=4)
X_reduced_KM4=kmeans4.fit_transform(X_reduced)
kmeans5=KMeans(n_clusters=5)
X_reduced_KM5=kmeans5.fit_transform(X_reduced)
kmeans6=KMeans(n_clusters=6)
X_reduced_KM6=kmeans6.fit_transform(X_reduced)
kmeans7=KMeans(n_clusters=7)
X_reduced_KM7=kmeans7.fit_transform(X_reduced)

#plt.scatter(X_reduced_KM[:,0],X_reduced_KM[:,1],c=y)
#plt.scatter(X[:,0],X[:,1])


plt.subplot(1,3,3)
#plt.scatter(xX,yX,c='g')
#plt.scatter(X_reduced_KM2[:,0],X_reduced_KM2[:,1],c=y)
#plt.show()
#plt.scatter(X_reduced_KM3[:,0],X_reduced_KM3[:,1],c=y)
#plt.show()
#plt.scatter(X_reduced_KM4[:,0],X_reduced_KM4[:,1],c=y)
#plt.show()
#plt.scatter(X_reduced_KM5[:,0],X_reduced_KM5[:,1],c=y)
#plt.show()
#plt.scatter(X_reduced_KM6[:,0],X_reduced_KM6[:,1],c=y)
#plt.show()
#plt.scatter(X_reduced_KM7[:,0],X_reduced_KM7[:,1],c=y)
#plt.show()


#plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
#plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
#plt.show()
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y)
plt.show()

#fig = plt.figure(1, figsize=(8, 6))
#ax = Axes3D(fig)
#ax.scatter(X_reduced_KM[:, 2], X_reduced_KM[:, 0], X_reduced_KM[:, 1], c=y)
#plt.show()

#3. Kwantyzacja
#3.1
obrazek=mpimg.imread('kolory.jpg')
print(obrazek.shape)

#3.2
img=np.array(obrazek).reshape(obrazek.shape[0]*obrazek.shape[1],obrazek.shape[2])
print('img',img)

wektoryzacja_red=img[:,0]
wektoryzacja_green=img[:,1]
wektoryzacja_blue=img[:,2]

#3.3
kmeans=KMeans(n_clusters=5).fit(img)
labels=kmeans.fit_predict(img)
print(kmeans.cluster_centers_)
print(labels)

for i in range(0,len(labels),1):
    wektoryzacja_red[i]=labels[i]
    wektoryzacja_green[i]=labels[i]
    wektoryzacja_blue[i]=labels[i]

#3.4
img_quant=np.vstack((wektoryzacja_red,wektoryzacja_green,wektoryzacja_blue)).T
print(img_quant)

#3.5
przeksztalcony_img_quant=np.reshape(img_quant,(obrazek.shape[0],obrazek.shape[1],obrazek.shape[2]))
print(przeksztalcony_img_quant.shape)

#3.6
obraz=plt.imshow(obrazek)
plt.title('Przed wektoryzacją')
plt.show()
cos=plt.imshow(przeksztalcony_img_quant)
plt.title('Po wektoryzacją')
plt.show()

#3.7

#3.8
def Wektoryzacja_n(obraz,n):
    img=np.reshape(obraz,(int(640*(480/2**n)),int(3*2**n)))
    x=plt.imshow(img)
    plt.show()

Wektoryzacja_n(img_quant,1)
Wektoryzacja_n(img_quant,3)
Wektoryzacja_n(img_quant,7)
Wektoryzacja_n(img_quant,9)
#Zwiekszanie parametru powoduje, ze obraz staje sie szerszy. Przy zbyt malym lub przy zbyt duzym parametrze n obraz staje sie niewidoczny

#3.9
perm=permutation(img_quant)
new_img=np.reshape(img,(640*480,3))[perm,:]
print(new_img)