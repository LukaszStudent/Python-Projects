import numpy as np
import scipy as sp
import pandas as pd
import math as m
import collections 

zoo = pd.read_csv('zoo.csv')

def get_index_positions_2(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    for i in range(len(list_of_elems)):
        if list_of_elems[i] == element:
            index_pos_list.append(i)
    return index_pos_list


#zadanie 1
def freq(x,prob=True):
    słownik={'kolumna':zoo.iloc[:,x],'kategoria':zoo.columns[0:len(zoo.columns)]}
    wartosc,liczebnosc=np.unique(słownik['kolumna'],return_counts=True)

    return wartosc,liczebnosc/len(słownik['kolumna'])

[xi,ni]=freq(15)
print('Zadanie 1')
print('Unikalne wartosci',xi,'\nEstymowane prawdopodobienstwo lub czestosc',ni)

#zadanie 2
def freq2(x,y,prob=True):
    słownik={'kolumna_x':zoo.iloc[:,x],'kolumna_y':zoo.iloc[:,y],'kategoria':zoo.columns[0:len(zoo.columns)]}
    wartosc_x,liczebnosc_x=np.unique(słownik['kolumna_x'],return_counts=True)
    wartosc_y,liczebnosc_y=np.unique(słownik['kolumna_y'],return_counts=True)
    lista_x=[]
    lista_y=[]
    for i in słownik['kolumna_x']:
        if i==prob:
            lista_x.append(1) #wartosci true dla kolumn x
        else:
            lista_x.append(2) #wartosci fals dla kolumny x
    for i in słownik['kolumna_y']:
        if i==prob:
            lista_y.append(3) #wartosci true dla kolumn y
        else:
            lista_y.append(4) #wartosci true dla kolumn y

    pozycja_xTRUE=get_index_positions_2(lista_x,1)
    pozycja_xFALSE=get_index_positions_2(lista_x,2)
    pozycja_yTRUE=get_index_positions_2(lista_y,3)
    pozycja_yFALSE=get_index_positions_2(lista_y,4)

    TxTy=pozycja_xTRUE+pozycja_yTRUE
    #print('TRUEx TRUE y',[item for item, count in collections.Counter(TxTy).items() if count > 1])
    liczebnoscTxTy=len([item for item, count in collections.Counter(TxTy).items() if count > 1])
    #print('True x True y',liczebnoscTxTy)

    TxFy=pozycja_xTRUE+pozycja_yFALSE
    #print('TRUEx FALSE y',[item for item, count in collections.Counter(TxFy).items() if count > 1])
    liczebnoscTxFy=len([item for item, count in collections.Counter(TxFy).items() if count > 1])
    #print('True x False y',liczebnoscTxFy)

    FxTy=pozycja_xFALSE+pozycja_yTRUE
    #print('FALSEx TRUE y',[item for item, count in collections.Counter(FxTy).items() if count > 1])
    liczebnoscFxTy=len([item for item, count in collections.Counter(FxTy).items() if count > 1])
    #print('False x True y',liczebnoscFxTy)

    FxFy=pozycja_xFALSE+pozycja_yFALSE
    #print('FALSEx FALSE y',[item for item, count in collections.Counter(FxFy).items() if count > 1])
    liczebnoscFxFy=len([item for item, count in collections.Counter(FxFy).items() if count > 1])
    #print('False x False y',liczebnoscFxFy)

    ni=[liczebnoscFxFy/len(słownik['kolumna_x']),liczebnoscFxTy/len(słownik['kolumna_x']),liczebnoscTxFy/len(słownik['kolumna_x']),liczebnoscTxTy/len(słownik['kolumna_x'])]

    #p(x) i p(y)
    #TxTFy=(liczebnoscTxTy+liczebnoscTxFy)/len(słownik['kolumna_x'])
    #FxTFy=(liczebnoscFxTy+liczebnoscFxFy)/len(słownik['kolumna_x'])
    #TyTFx=(liczebnoscTxTy+liczebnoscFxTy)/len(słownik['kolumna_x'])
    #FyTFx=(liczebnoscTxFy+liczebnoscFxFy)/len(słownik['kolumna_x'])
    #print(TxTFy,FxTFy,TyTFx,FyTFx)
    #ni=[TxTFy,FxTFy,TyTFx,FyTFx]

    return wartosc_x,wartosc_y,ni

[xi2,yi2,ni]=freq2(3,15)
print('\nZadanie 2')
print('Unikalne wartosci xi',xi2,'unikalne wartosci yi',yi2,'\nlaczny rozklad prawdopodobienstwa',ni)

#zadanie 3
def entropy(x):
    suma=0
    xi,pr=freq(x)
    for n in range(0,len(xi),1):
        suma+=(-1)*(pr[n]*m.log2(pr[n]))

    return suma

h=entropy(4)
print('\nZadanie 3')
print('Entropia H(X)',h)

#I(X,Y)
def infogain_XY(x,y):
    _,_,H_XY=freq2(x,y)
    HXY=0
    #H(X,Y)
    for i in range(0,len(H_XY),1):
        HXY+=(H_XY[i]*m.log2((H_XY[i])**(-1)))
    I_XY=entropy(x)+entropy(y)-HXY
    return I_XY

iXY=infogain_XY(3,15)
print('Informacja wzajemna I(X,Y)',iXY)

#I(Y,X)
def infogain_YX(x,y):
    H_YIX=0
    xi,Pr=freq(x)
    xi2,yi2,Pr2=freq2(x,y)

    HYIXxi = 0
    for i in range(len(Pr2)):
        for j in range(len(Pr)):
            HYIXxi += ((Pr2[i]/Pr[j])*m.log2((Pr2[i]/Pr[j])**(1)))

    I_YX=entropy(y)-H_YIX
    return I_YX

iYX=infogain_YX(3,15)
print('Przyrost informacji I(Y,X)',iYX)

#zadanie 6
#url='http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm'
#data=pd.read_csv(url)
#print(data)
