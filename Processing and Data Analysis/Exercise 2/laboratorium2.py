import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.DataFrame({"x":[1,2,3,4,5],"y": ['a','b','a','b','b']})

#zadanie 1
srednia=df.groupby('y')['x'].mean()
print(srednia)

#zadanie 2
liczebnosc=df.value_counts()
print(liczebnosc)

#zadanie 3
plik=np.loadtxt('autos.csv',dtype=str,delimiter=',')
print(plik)

plikPANDA=pd.read_csv('autos.csv')
print(plikPANDA)
#za pomoca PANDY dane sa bardziej przejrzyste niż za pomocą NumPy

#zadanie 4
plikPANDA=pd.read_csv('autos.csv')
auta=pd.DataFrame(plikPANDA)
spalanie=auta.groupby('make')['city-mpg','highway-mpg'].mean()
print(spalanie)

#zadanie 5
plikPANDA=pd.read_csv('autos.csv')
auta=pd.DataFrame(plikPANDA)
paliwo=auta.value_counts(subset=['make','fuel-type'])
print(paliwo)

#zadanie 6
plikPANDA=pd.read_csv('autos.csv')
auta=pd.DataFrame(plikPANDA)
x=auta['city-mpg']
y=auta['length']
wielomian1PF=np.polyfit(x,y,1)
wielomian2PF=np.polyfit(x,y,2)
print('Wielomian 1-ego st. np.polyfit ' + str(wielomian1PF))
print('Wielomian 2-ego st. np.polyfit ' + str(wielomian2PF)+'\n')

wielomian1PV1PF=np.polyval(wielomian1PF,1)
wielomian2PV1PF=np.polyval(wielomian1PF,2)
wielomian1PV2PF=np.polyval(wielomian2PF,1)
wielomian2PV2PF=np.polyval(wielomian2PF,2)
print('Wielomian 1-ego st. np.polyval wzgledem wielomianem 1-ego st. np.polyfit ' + str(wielomian1PV1PF))
print('Wielomian 2-ego st. np.polyval wzgledem wielomianem 1-ego st. np.polyfit ' + str(wielomian2PV1PF))
print('Wielomian 1-ego st. np.polyval wzgledem wielomianem 2-ego st. np.polyfit ' + str(wielomian1PV2PF))
print('Wielomian 2-ego st. np.polyval wzgledem wielomianem 2-ego st. np.polyfit ' + str(wielomian2PV2PF))

#zadanie 7
plikPANDA=pd.read_csv('autos.csv')
auta=pd.DataFrame(plikPANDA)
x=auta['city-mpg']
y=auta['length']
korelacja=x.corr(y)
print(korelacja)

##zadanie 8
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Próbki')
ax.plot(x, korelacja * x, label='Korelacja')
ax.set_xlabel('city-mpg')
ax.set_ylabel('length')
ax.legend(facecolor='white')
plt.show()

#zadanie 9
plikPANDA=pd.read_csv('autos.csv')
auta=pd.DataFrame(plikPANDA)
zmienna=auta['length']
estymator=stats.gaussian_kde(zmienna)
x=np.linspace(zmienna.min()-1,zmienna.max()+1,len(zmienna))
y=estymator(x)

plt.plot(x,y,'k')
plt.plot(zmienna,y,'b*')
plt.show()

#zadanie 10
plikPANDA=pd.read_csv('autos.csv')
auta=pd.DataFrame(plikPANDA)
zmienna=auta['length']
estymator=stats.gaussian_kde(zmienna)
x=np.linspace(zmienna.min()-1,zmienna.max()+1,len(zmienna))
y=estymator(x)
plt.subplot(2,1,1)
plt.plot(x,y,'k')
plt.plot(zmienna,y,'g*')

zmienna=auta['width']
estymator=stats.gaussian_kde(zmienna)
x=np.linspace(zmienna.min()-1,zmienna.max()+1,len(zmienna))
y=estymator(x)
plt.subplot(2,1,2)
plt.plot(x,y,'k')
plt.plot(zmienna,y,'r*')
plt.show()
