import math as m
import matplotlib.pyplot as plt
from scipy import fftpack
import time

kolejne_czasy_DFT=[]
kolejne_czasy_FFT=[]

lrzeczywista,lurojona,tab,czas=[],[],[],[]
def DFT(tablica):
    for k in range(0,N,1):
        sumarzecz,sumaurojona=0,0
        for n in range(0,N,1):
            rzeczywista=tablica[n]*m.cos((-2*m.pi*k*n)/N)
            urojona=tablica[n]*m.sin((-2*m.pi*k*n)/N)
            sumarzecz+=rzeczywista
            sumaurojona-=urojona
        lrzeczywista.append(sumarzecz)
        lurojona.append(sumaurojona)


lmodul,ltg,lalfa,index=[],[],[],[]
def WidmoAmplitudowe():
    for k in range(0,int(N/2),1):
        index.append(k)
        modul=m.sqrt((lrzeczywista[k])**2+(lurojona[k])**2)
        lmodul.append(modul)
        tg=lurojona[k]/lrzeczywista[k]
        ltg.append(tg)
        #alfa=m.atan(lurojona[k]/rzeczywista[k])
        #lalfa.append(alfa)
    #plt.plot(index,lmodul)
    #plt.title('Widmo')
    #plt.ylabel('modul')
    #plt.xlabel('index')
    #plt.show()


modul_prim=[]
def WidmoDecybel():
    for k in range(0,int(N/2),1):
        modul=10*m.log10(lmodul[k])
        modul_prim.append(modul)

    plt.plot(modul_prim)
    plt.ylabel('A [dB]')
    plt.xlabel('f [Hz]')
    #plt.show()


lfk=[]
def CzestProzkowania(N):
    for k in range(0,int(N/2),1):
        fk=k*(fs/N)
        lfk.append(fk)
    #plt.plot(lfk,lmodul)
    #plt.show()


#zadanie 4
#funkcja nr 14 z tabeli nr 4
zadanie4_start=time.time()
Tc=1
fs=22050
f=11025 #czestotliwosc
Ts=1/fs #okres probkowania
N=round(Tc*fs,0) #liczba probek na caly sygnal
lista1,lista2,lista3=[],[],[]
czas1,czas2,czas3=[],[],[]
suma1,suma2,suma3=0,0,0
for n in range(0,N,1):
    t=n*Ts
    for h in range(1,3,1):
        b=(m.sin((h*t*m.pi)/2+m.cos(h**2*m.pi*t)))
        suma1+=b
        lista1.append(suma1)
        czas1.append(t)

    for h in range(1,7,1):
        b=(m.sin((h*t*m.pi)/2+m.cos(h**2*m.pi*t)))
        suma2+=b
        lista2.append(suma2)
        czas2.append(t)

    for h in range(1,11,1):
        b=(m.sin((h*t*m.pi)/2+m.cos(h**2*m.pi*t)))
        suma3+=b
        lista3.append(suma3)
        czas3.append(t)


#print('zadanie 4 1')
#zadanie4_1_start=time.time()
DFT(lista1)
zadanie4_1_koniec=time.time()
WidmoAmplitudowe()
plt.subplot(1,2,1)
WidmoDecybel()
plt.subplot(1,2,2)
WidmoDecybel()
plt.xlim(11015,11025)
plt.show()
CzestProzkowania(N)
zadanie4_1_czas=zadanie4_1_koniec-zadanie4_1_start
kolejne_czasy_DFT.append(zadanie4_1_czas)
print('Czas wykonania DFT dla pierwszego przedzia³u w zadaniu 4 wynosci: ',zadanie4_1_czas)
lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear()

print('zadanie 4 2')
zadanie4_2_start=time.time()
DFT(lista2)
zadanie4_2_koniec=time.time()
WidmoAmplitudowe()
plt.subplot(2,2,1)
WidmoDecybel()
plt.subplot(2,2,2)
WidmoDecybel()
plt.xlim(3660,3690)
plt.subplot(2,2,3)
WidmoDecybel()
plt.xlim(7330,7370)
plt.subplot(2,2,4)
WidmoDecybel()
plt.xlim(11010,11025)
plt.show()
CzestProzkowania(N)
zadanie4_2_czas=zadanie4_2_koniec-zadanie4_2_start
kolejne_czasy_DFT.append(zadanie4_2_czas)
print('Czas wykonania DFT dla drugiego przedzia³u w zadaniu 4 wynosci: ',zadanie4_2_czas)
lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear(),czas.clear()

print('zadanie 4 3')
zadanie4_3_start=time.time()
DFT(lista3)
zadanie4_3_koniec=time.time()
WidmoAmplitudowe()
plt.subplot(2,3,1)
WidmoDecybel()
plt.subplot(2,3,2)
WidmoDecybel()
plt.xlim(2180,2230)
plt.subplot(2,3,3)
WidmoDecybel()
plt.xlim(4380,4440)
plt.subplot(2,3,4)
WidmoDecybel()
plt.xlim(6590,6640)
plt.subplot(2,3,5)
WidmoDecybel()
plt.xlim(8800,8840)
plt.subplot(2,3,6)
WidmoDecybel()
plt.xlim(11000,11025)
plt.show()
CzestProzkowania(N)
zadanie4_3_czas=zadanie4_3_koniec-zadanie4_3_start
kolejne_czasy_DFT.append(zadanie4_3_czas)
print('Czas wykonania DFT dla trzeciego przedzia³u w zadaniu 4 wynosci: ',zadanie4_3_czas)


zad4_1_start=time.time()
zad4_1=fftpack.fft(lista1)
zad4_1_koniec=time.time()
zad4_1_czas=zad4_1_koniec-zad4_1_start
kolejne_czasy_FFT.append(zad4_1_czas)
print('Czas wykonania FFT dla zad 4 1-przedzial: ',zad4_1_czas)


zad4_2_start=time.time()
zad4_2=fftpack.fft(lista2)
zad4_2_koniec=time.time()
zad4_2_czas=zad4_2_koniec-zad4_2_start
kolejne_czasy_FFT.append(zad4_2_czas)
print('Czas wykonania FFT dla zad 4 2-przedzial: ',zad4_2_czas)


zad4_3_start=time.time()
zad4_3=fftpack.fft(lista3)
zad4_3_koniec=time.time()
zad4_3_czas=zad4_3_koniec-zad4_3_start
kolejne_czasy_FFT.append(zad4_3_czas)
print('Czas wykonania FFT dla zad 4 3-przedzial: ',zad4_3_czas)
print(kolejne_czasy_FFT)
print(kolejne_czasy_DFT)