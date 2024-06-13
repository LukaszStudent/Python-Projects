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

#zadanie 1
zadanie1_start=time.time()
print('zadanie 1')
Tc=1 #czas trwania sygnalu
fs=16000 #czestotliwosc probkowania fs>=2*fmax #zmienione z 20000
f=8000 #czestotliwosc #zmienione z 10000
Ts=1/fs #okres probkowania
N=round(Tc*fs,0) #liczba probek na caly sygnal
tabx=[]
for n in range(0,N,1):
    t=n*Ts
    x=((m.exp(-t)*m.sin(m.pi*f*t))/(2.0001+m.cos(m.pi*t)))
    tabx.append(x)
    czas.append(t)

DFT(tabx)
zadanie1_koniec=time.time()
WidmoAmplitudowe()
plt.subplot(1,2,1)
WidmoDecybel()
plt.subplot(1,2,2)
WidmoDecybel()
plt.xlim(3950,4050)
plt.show()
CzestProzkowania(N)

zadanie1_czas=zadanie1_koniec-zadanie1_start
print('Czas wykonania DFT dla zadanie 1 wynosci: ',zadanie1_czas)
kolejne_czasy_DFT.append(zadanie1_czas)

zad1_start=time.time()
zad1=fftpack.fft(tabx)
zad1_koniec=time.time()
zad1_czas=zad1_koniec-zad1_start
kolejne_czasy_FFT.append(zad1_czas)
print('Czas wykonania FFT dla zad 1: ',zad1_czas)
print(kolejne_czasy_FFT)
print(kolejne_czasy_DFT)