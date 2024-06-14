from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import sklearn.datasets # probki badawcze
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from matplotlib.colors import ListedColormap

#1.1
X,y= sklearn.datasets.make_classification(n_samples=100)

#1.2
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

#1.3
clf_Gaussian=GaussianNB()
clf_Quadratic=QuadraticDiscriminantAnalysis()
clf_KNeighbours=KNeighborsClassifier()
clf_SVC=SVC(probability=True)
clf_DTC=DecisionTreeClassifier()

DataFrame=pd.DataFrame(columns=['as_Gaussian','as_Quadratic','as_KNeighbours','as_SVC','as_DTC',...
                                ,'rs_Gaussian','rs_Quadratic','rs_KNeighbours','rs_SVC','rs_DTC',...
                                ,'ps_Gaussian','ps_Quadratic','ps_KNeighbours','ps_SVC','ps_DTC',...
                                ,'f1s_Gaussian','f1s_Quadratic','f1s_KNeighbours','f1s_SVC','f1s_DTC',...
                                ,'roc_Gaussian','roc_Quadratic','roc_KNeighbours','roc_SVC','roc_DTC'])
print(DataFrame)


for i in range(0,100,1):
    X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y)
    cGf=clf_Gaussian.fit(X_train,y_train)
    cQf=clf_Quadratic.fit(X_train,y_train)
    cKNf=clf_KNeighbours.fit(X_train,y_train)
    cSf=clf_SVC.fit(X_train,y_train)
    cDf=clf_DTC.fit(X_train,y_train)

    y_pred_Gaussian=cGf.predict(X_test)
    y_pred_Quadratic=cQf.predict(X_test)
    y_pred_KNeighbours=cKNf.predict(X_test)
    y_pred_SVC=cSf.predict(X_test)
    y_pred_DTC=cDf.predict(X_test)

    dokladnosc_Gaussian=sklearn.metrics.accuracy_score(y_test,y_pred_Gaussian)
    dokladnosc_Quadratic=sklearn.metrics.accuracy_score(y_test,y_pred_Quadratic)
    dokladnosc_KNeighbours=sklearn.metrics.accuracy_score(y_test,y_pred_KNeighbours)
    dokladnosc_SVC=sklearn.metrics.accuracy_score(y_test,y_pred_SVC)
    dokladnosc_DTC=sklearn.metrics.accuracy_score(y_test,y_pred_DTC)

    czulosc_Gaussian=sklearn.metrics.recall_score(y_test,y_pred_Gaussian)
    czulosc_Quadratic=sklearn.metrics.recall_score(y_test,y_pred_Quadratic)
    czulosc_KNeighbours=sklearn.metrics.recall_score(y_test,y_pred_KNeighbours)
    czulosc_SVC=sklearn.metrics.recall_score(y_test,y_pred_SVC)
    czulosc_DTC=sklearn.metrics.recall_score(y_test,y_pred_DTC)

    precyzja_Gaussian=sklearn.metrics.precision_score(y_test,y_pred_Gaussian)
    precyzja_Quadratic=sklearn.metrics.precision_score(y_test,y_pred_Quadratic)
    precyzja_KNeighbours=sklearn.metrics.precision_score(y_test,y_pred_KNeighbours)
    precyzja_SVC=sklearn.metrics.precision_score(y_test,y_pred_SVC)
    precyzja_DTC=sklearn.metrics.precision_score(y_test,y_pred_DTC)

    F1_Gaussian=sklearn.metrics.f1_score(y_test,y_pred_Gaussian)
    F1_Quadratic=sklearn.metrics.f1_score(y_test,y_pred_Quadratic)
    F1_KNeighbours=sklearn.metrics.f1_score(y_test,y_pred_KNeighbours)
    F1_SVC=sklearn.metrics.f1_score(y_test,y_pred_SVC)
    F1_DTC=sklearn.metrics.f1_score(y_test,y_pred_DTC)

    pole_pod_krzywa_auc_Gaussian=sklearn.metrics.roc_auc_score(y_test,y_pred_Gaussian)
    pole_pod_krzywa_auc_Quadratic=sklearn.metrics.roc_auc_score(y_test,y_pred_Quadratic)
    pole_pod_krzywa_auc_KNeighbours=sklearn.metrics.roc_auc_score(y_test,y_pred_KNeighbours)
    pole_pod_krzywa_auc_SVC=sklearn.metrics.roc_auc_score(y_test,y_pred_SVC)
    pole_pod_krzywa_auc_DTC=sklearn.metrics.roc_auc_score(y_test,y_pred_DTC)

    DataFrame=DataFrame.append({'as_Gaussian':dokladnosc_Gaussian,'as_Quadratic':dokladnosc_Quadratic,'as_KNeighbours':dokladnosc_KNeighbours,'as_SVC':dokladnosc_SVC,'as_DTC':dokladnosc_DTC
                                ,'rs_Gaussian':czulosc_Gaussian,'rs_Quadratic':czulosc_Quadratic,'rs_KNeighbours':czulosc_KNeighbours,'rs_SVC':czulosc_SVC,'rs_DTC':czulosc_DTC
                                ,'ps_Gaussian':precyzja_Gaussian,'ps_Quadratic':precyzja_Quadratic,'ps_KNeighbours':precyzja_KNeighbours,'ps_SVC':precyzja_SVC,'ps_DTC':precyzja_DTC
                                ,'f1s_Gaussian':F1_Gaussian,'f1s_Quadratic':F1_Quadratic,'f1s_KNeighbours':F1_KNeighbours,'f1s_SVC':F1_SVC,'f1s_DTC':F1_DTC
                                ,'roc_Gaussian':pole_pod_krzywa_auc_Gaussian,'roc_Quadratic':pole_pod_krzywa_auc_Quadratic,'roc_KNeighbours':pole_pod_krzywa_auc_KNeighbours,'roc_SVC':pole_pod_krzywa_auc_SVC,'roc_DTC':pole_pod_krzywa_auc_DTC},ignore_index=True)
   
    if i==99:
        #h = .02
        #x_min, x_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
        #y_min, y_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
        #xx, yy = np.meshgrid(np.arange(x_min, x_max,h),np.arange(y_min, y_max,h))
        #Z = clf_KNeighbours.predict(np.c_[xx.ravel(), yy.ravel()])
        #Z = Z.reshape(xx.shape)
        #plt.contourf(xx, yy,cmap=plt.cm.Paired)
        #plt.scatter(X[:, 0],X[:, 1],c=y, alpha=1.0, edgecolor="black")
        #plt.show()

        
        plt.subplot(1,3,1)
        plt.title('Oczekiwane')
        plt.scatter(X_test[:,0],X_test[:,1],c=y_test)
        plt.subplot(1,3,2)
        plt.title('Obliczone')
        plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
        plt.subplot(1,3,3)
        plt.title('Różnice')
        plt.scatter(X_train[:,0],X_train[:,1],c='g')
        plt.scatter(X_test[:,0],X_test[:,1],c='r')
        plt.show()

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test,y_pred_KNeighbours)
        print(fpr, tpr, thresholds)
        plt.plot(fpr,tpr)
        plt.plot((0,1),(0,1),'r--')
        plt.title('Krzywa ROC przy pomocy FPR i TPR')
        plt.show()

        ax=sklearn.metrics.plot_roc_curve(clf_KNeighbours,X_test,y_test)
        plt.title('Krzywa ROC przy pomocy funkcji "lot_roc_curve"')
        plt.plot((0,1),(0,1),'r--')
        plt.show()

#print(DataFrame)

nazwy=['accuracy_score','recall_score','precision_score','f1_score','roc_auc']

wartosci=DataFrame.mean().values
#print(wartosci)

Gaussian_wartosci=wartosci[::6]
Quadratic_wartosci=wartosci[1::6]
KNeigbours_wartosci=wartosci[2::6]
SVC_wartosci=wartosci[3::6]
DTC_wartosci=wartosci[4::6]

DataFrame_mean=pd.DataFrame({'Gaussian':Gaussian_wartosci,'Quadratic':Quadratic_wartosci,'KNeighbours':KNeigbours_wartosci,'SVC':SVC_wartosci,'DTC':DTC_wartosci},index=nazwy)
ax=DataFrame_mean.plot.bar(rot=0)
plt.show()


#2. Badanie parametrow wybrangeo klasyfikatora
#2.1
X,y = sklearn.datasets.make_classification(n_samples=200,n_classes=2)
#print(X,y)
#2.2
svc=SVC(probability=True)
KN=KNeighborsClassifier()

#2.3
slownik_KNeighbours={'n_neighbors':[1,2,3,4,5,6,7,8,9,10],'weights':('uniform','distance'),'algorithm':('auto','ball_tree','kd_tree','brute'),'p':[1,2]}
slownik_KNeighbours_najwazniejsze={'n_neighbors':[1,2,3,4,5],'p':[1,2,3,4]}

#2.4
clf = GridSearchCV(KN, slownik_KNeighbours_najwazniejsze)
clf.fit(X,y)
#print(clf.cv_results_)
#print(sorted(clf.cv_results_.keys()))

DataFrame=pd.DataFrame(clf.cv_results_)
pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(DataFrame)
DataFrame_mean_test_score=DataFrame[['mean_test_score']]
pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(DataFrame)
ax=DataFrame_mean_test_score.plot(rot=0)
plt.show()
najlepsze_wartosci=clf.best_params_
#print(najlepsze_wartosci['n_neighbors'])
#print(najlepsze_wartosci['p'])
#DataFrame=DataFrame[['param_n_neighbors','param_p']]
#plt.pcolormesh(DataFrame[['param_n_neighbors']],DataFrame[['param_p']])
#plt.show()


clf=KNeighborsClassifier(n_neighbors=najlepsze_wartosci['n_neighbors'],p=najlepsze_wartosci['p'])
#2.6

DataFrame=pd.DataFrame(columns=['as_clf','rs_clf','ps_clf','f1_clf','roc_clf'])

for i in range(0,100,1):
    X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y)
    cflfit=clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    dokladnosc_clf=sklearn.metrics.accuracy_score(y_test,y_pred)
    czulosc_clf=sklearn.metrics.recall_score(y_test,y_pred)
    precyzja_clf=sklearn.metrics.precision_score(y_test,y_pred)
    F1_clf=sklearn.metrics.f1_score(y_test,y_pred)
    pole_pod_krzywa_auc_clf=sklearn.metrics.roc_auc_score(y_test,y_pred)

    DataFrame=DataFrame.append({'as_clf':dokladnosc_clf,'rs_clf':czulosc_clf,
                                'ps_clf':precyzja_clf,'f1_clf':F1_clf,'roc_clf':pole_pod_krzywa_auc_clf},ignore_index=True)

    if i==99:
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test,y_pred)
        print(fpr, tpr, thresholds)
        plt.plot(fpr,tpr)
        plt.plot((0,1),(0,1),'r--')
        plt.title('Krzywa ROC przy pomocy FPR i TPR')
        plt.show()

        ax=sklearn.metrics.plot_roc_curve(clf_KNeighbours,X_test,y_test)
        plt.title('Krzywa ROC przy pomocy funkcji "lot_roc_curve"')
        plt.plot((0,1),(0,1),'r--')
        plt.show()

#print(DataFrame)

DataFrame=DataFrame.mean()
#print(DataFrame)
                            
