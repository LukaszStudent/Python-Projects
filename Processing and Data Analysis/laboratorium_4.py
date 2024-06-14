import math as m
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage


# 2.Dyskretyzacja

def Dyskretyzacja(x:'Czestotliwosc probkowania'):
    Tc=1 #czas trwania sygnalu
    fs=x #czestotliwosc probkowania fs>=2*fmax
    f=10 #czestotliwosc
    Ts=1/fs #okres probkowania
    N=round(Tc*fs,0) #liczba probek na caly sygnal
    tab=[]
    czas=[]
    for n in range(0,N,1):
        t=n*Ts
        fun=(m.sin(2*m.pi*f*t))
        tab.append(fun)
        czas.append(t)
    plt.title('fs='+str(x)+'Hz')
    plt.plot(czas,tab)

plt.subplot(4,3,1)
Dyskretyzacja(20)

plt.subplot(4,3,2)
Dyskretyzacja(21)

plt.subplot(4,3,3)
Dyskretyzacja(30)

plt.subplot(4,3,4)
Dyskretyzacja(45)

plt.subplot(4,3,5)
Dyskretyzacja(50)

plt.subplot(4,3,6)
Dyskretyzacja(100)

plt.subplot(4,3,7)
Dyskretyzacja(150)

plt.subplot(4,3,8)
Dyskretyzacja(200)

plt.subplot(4,3,9)
Dyskretyzacja(250)

plt.subplot(4,3,11)
Dyskretyzacja(1000)
plt.show()

#zadanie 2.4 
#Istenije takie twierdzenie - Twierdzenie o probkowaniu 

#zadanie 2.5
#Aliasing

#################################################################################

# 3.Kwantyzacja

#zadanie 3.1
obrazek = mpimg.imread('zad3.jpg')
plt.subplot(2,5,1)
plt.title('Oryginał')
plt.imshow(obrazek)


#zadanie 3.2
szerokosc,wysokosc,_=obrazek.shape
print('Szerokosc: '+str(szerokosc))
print('Wysokosc: '+str(wysokosc))

#zadanie 3.3
_,_,wartosc=obrazek.shape
print('Pojedynczy piksel opisywany jest: '+str(wartosc)+' wartosciami')

#zadanie 3.4
jasnosc=(obrazek.max()+obrazek.min())/2
jasnosc=jasnosc/1000
bright= lambda rgb : np.dot(rgb[... , :3] , [jasnosc,jasnosc,jasnosc]) 
jasnosc_piksela_szary=bright(obrazek)
plt.subplot(2,5,2)
plt.title('Jasnosc piksela')
plt.imshow(jasnosc_piksela_szary,cmap=plt.get_cmap(name='gray'))



srednia1=np.average(obrazek,axis=(0,1))
print(srednia1)
sredni_piksel= lambda rgb : np.dot(rgb[... , :3] , srednia1/1000) 
sredni_szary=sredni_piksel(obrazek)
print(sredni_szary)
plt.subplot(2,5,3)
plt.title('Sr. wart. piksela')
plt.imshow(sredni_szary,cmap=plt.get_cmap(name='gray'))


iluminacja = lambda rgb : np.dot(rgb[... , :3] , [0.21 , 0.72, 0.07]) 
szary_iluminacja = iluminacja(obrazek)
plt.subplot(2,5,4)
plt.title('Iluminacja')
plt.imshow(szary_iluminacja,cmap=plt.get_cmap(name='gray'))

#zadanie 3.5
histogram_obrazek = np.histogram(obrazek,bins=255)
histogram_jasnosc_piksela =np.histogram(jasnosc_piksela_szary,bins=255)
histogram_sredni_szary = np.histogram(sredni_szary,bins=255)
histogram_iluminacja = np.histogram(szary_iluminacja,bins=255)
histogram_szary16 = np.histogram(sredni_szary,bins=16)

plt.subplot(2,5,6)
plt.title('Hist oryginał')
plt.bar(histogram_obrazek[1][:-1],histogram_obrazek[0])

plt.subplot(2,5,7)
plt.title('Hist jasnosc piksela')
plt.bar(histogram_jasnosc_piksela[1][:-1],histogram_jasnosc_piksela[0])

plt.subplot(2,5,8)
plt.title('Hist sr. wart. piksela')
plt.bar(histogram_sredni_szary[1][:-1],histogram_sredni_szary[0])

plt.subplot(2,5,9)
plt.title('Hist iluminacji')
plt.bar(histogram_iluminacja[1][:-1],histogram_iluminacja[0])

plt.subplot(2,5,10)
plt.title('Hist 16 bins')
plt.bar(histogram_szary16[1][:-1],histogram_szary16[0])
plt.xlim
plt.show()



# 4.Binaryzacja

#zadanie 4.1
gradient = mpimg.imread('gradient.jpg')
cos=gradient
plt.title('Oryginał')
plt.imshow(gradient)
plt.show()

#zadanie 4.2
szary_gradient=iluminacja(gradient)
plt.title('Przekształcenie za pomocą iluminacji')
plt.imshow(szary_gradient,cmap=plt.get_cmap(name='gray'))
plt.show()

#zadanie 4.3
histogram_gradient, krawedzie = np.histogram(szary_gradient,bins=255,range=(0,1))
plt.title('Histogram')
plt.plot(krawedzie[0:-1],histogram_gradient,label='Histogram')
minimum=min(histogram_gradient)
i,=np.where(histogram_gradient==minimum)
prog=krawedzie[i]

plt.axvline(x=prog,color='r',linestyle='-',label='Próg')
plt.legend(loc='upper right')
plt.show()

#zadanie 4.4
szary_gradient[szary_gradient>=prog]=1
szary_gradient[szary_gradient<prog]=0
gradient[:,:,0]=szary_gradient
gradient[:,:,1]=szary_gradient
gradient[:,:,2]=szary_gradient
plt.imshow(gradient)
plt.show()
