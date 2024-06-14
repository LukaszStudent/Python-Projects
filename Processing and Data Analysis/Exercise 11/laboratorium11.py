import numpy as np
import pandas as pd
import sklearn.datasets # probki badawcze
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import label_binarize

X, y = sklearn.datasets.make_classification(n_samples=2000,n_features=2,n_classes=4,n_clusters_per_class=1,n_redundant=0,n_repeated=0,random_state=3)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,train_size=0.5)

SVC_linear=SVC(kernel='linear',probability=True)
SVC_rbf=SVC(kernel='rbf',probability=True)
LG=LogisticRegression()
Per=Perceptron()

klasyfikatory=[SVC_linear,SVC_rbf,LG,Per]

kolumny=['as_SVC_linear','rs_SVC_linear','ps_SVC_linear','f1s_SVC_linear',
'as_SVC_rbf','rs_SVC_rbf','ps_SVC_rbf','f1s_SVC_rbf',
'as_LG','rs_LG','ps_LG','f1s_LG',
'as_Per','rs_Per','ps_Per','f1s_Per']


print('OvO')
wartosci_OvO=[]
for i in klasyfikatory:
    print(i)
    OvO=OneVsOneClassifier(i)
    clf=OvO.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    dokladnosc=metrics.accuracy_score(y_test,y_pred)
    print('Accuracy score',dokladnosc)
    czulosc=metrics.recall_score(y_test,y_pred,average='macro')
    print('Recall score',czulosc)
    precyzja=metrics.precision_score(y_test,y_pred,average='macro')
    print('Precicion score',precyzja)
    f1=metrics.f1_score(y_test,y_pred,average='macro')
    print('F1 score',f1,'\n')
    #if i!=Per:
    #    y_pred_proba=clf.decision_function(X_test)
    #    print(y_pred_proba)
    #    pole_pod_krzywa=sklearn.metrics.roc_auc_score(y_test,y_pred_proba,multi_class='ovo')
    #    print(pole_pod_krzywa)
    #else:
    #    y_pred_proba=clf.score(X_test,y_test)
    #    pole_pod_krzywa=sklearn.metrics.roc_auc_score(y_test,y_pred_proba,multi_class='ovo')
    #    print(pole_pod_krzywa)

    wartosci_OvO.append(dokladnosc)
    wartosci_OvO.append(czulosc)
    wartosci_OvO.append(precyzja)
    wartosci_OvO.append(f1)


    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,alpha=0.6)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.plot([],[],color='purple',alpha=0.6,label='klasa 0', linewidth=5)
    plt.plot([],[],color='orange',alpha=0.6,label='klasa 1', linewidth=5)
    plt.plot([],[],color='blue',alpha=0.6,label='klasa 2', linewidth=5)
    plt.plot([],[],color='green',alpha=0.6,label='klasa 3', linewidth=5)
    plt.title('OvO {tytul:}'.format(tytul=i))
    plt.legend()
    plt.show()

    #Ladniejsze wykresy, ale dluzej sie tworza
    #plot_decision_regions(X_train,y_train,clf=OvO)
    #plt.show()


print('OvR')
wartosci_OvR=[]
for i in klasyfikatory:
    print(i)
    OvR=OneVsRestClassifier(i)
    clf=OvR.fit(X,y)
    y_pred=clf.predict(X_train)
    dokladnosc=metrics.accuracy_score(y_test,y_pred)
    print('Accuracy score',dokladnosc)
    czulosc=metrics.recall_score(y_test,y_pred,average='macro')
    print('Recall score',czulosc)
    precyzja=metrics.precision_score(y_test,y_pred,average='macro')
    print('Precicion score',precyzja)
    f1=metrics.f1_score(y_test,y_pred,average='macro')
    print('F1 score',f1)

    if i!=Per:
        y_pred_proba=clf.predict_proba(X_train)
        pole_pod_krzywa=sklearn.metrics.roc_auc_score(y_test,y_pred_proba,multi_class='ovr')
        print('Roc auc score',pole_pod_krzywa,'\n')


    wartosci_OvR.append(dokladnosc)
    wartosci_OvR.append(czulosc)
    wartosci_OvR.append(precyzja)
    wartosci_OvR.append(f1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,alpha=0.6)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.plot([],[],color='purple',alpha=0.6,label='klasa 0', linewidth=5)
    plt.plot([],[],color='orange',alpha=0.6,label='klasa 1', linewidth=5)
    plt.plot([],[],color='blue',alpha=0.6,label='klasa 2', linewidth=5)
    plt.plot([],[],color='green',alpha=0.6,label='klasa 3', linewidth=5)
    plt.title('OvR {tytul:}'.format(tytul=i))
    plt.legend()
    plt.show()

    #Ladniejsze wykresy, ale dluzej sie tworza
    #plot_decision_regions(X_train,y_train,clf=OvR)
    #plt.show()

data_OvO=[]
zipped_OvO=zip(kolumny,wartosci_OvO)
slownik_OvO=dict(zipped_OvO)

data_OvR=[]
zipped_OvR=zip(kolumny,wartosci_OvR)
slownik_OvR=dict(zipped_OvR)

DataFrame=pd.DataFrame()
DataFrame=DataFrame.append(slownik_OvO,True)
pd.set_option("display.max_rows", None, "display.max_columns", None)
DataFrame=DataFrame.append(slownik_OvR,True)

ac_OvO=DataFrame.iloc[0,0:4].values
f1_OvO=DataFrame.iloc[0,4:8].values
ps_OvO=DataFrame.iloc[0,8:12].values
rs_OvO=DataFrame.iloc[0,12:].values

ac_OvR=DataFrame.iloc[1,0:4].values
f1_OvR=DataFrame.iloc[1,4:8].values
ps_OvR=DataFrame.iloc[1,8:12].values
rs_OvR=DataFrame.iloc[1,12:].values

nazwy=['accuracy_score','recall_score','precision_score','f1_score']
DataFrame_wykres=pd.DataFrame({'OvO==SVC linear':ac_OvO,'OvO==SVC rbf':rs_OvO,'OvO==LogisticRegression':ps_OvO,'OvO==Perceptron':f1_OvO,'OvR==SVC linear':ac_OvR,'OvR==SVC rbf':rs_OvR,'OvR==LogisticRegression':ps_OvR,'OvR==Perceptron':f1_OvR},index=nazwy)#
ax=DataFrame_wykres.plot.bar(rot=0)
plt.show()

plt.subplot(1,3,1)
plt.title('Oczekiwane')
plt.scatter(X_test[:,0],X_test[:,1],c=y_test)
plt.subplot(1,3,2)
plt.title('Obliczone')
plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
plt.subplot(1,3,3)
plt.title('Różnice')
plt.scatter(X_train[:,0],X_train[:,1],c='r')
plt.scatter(X_test[:,0],X_test[:,1],c='g')
plt.show()


y = label_binarize(y, classes=[0, 1, 2,3])
n_classes = y.shape[1]
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5,train_size=0.5)

for i in klasyfikatory:
    print(i)
    classifier = OneVsRestClassifier(i)
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='Krzywa ROC (AUC = {0:0.2f}) dla klasy {1}'''.format(roc_auc[i],i))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC dla {0}'.format(classifier))
    plt.legend(loc="lower right")
    plt.show()
