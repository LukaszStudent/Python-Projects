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

#zadanie 3
#zadanie3_start=time.time()
Tc=1 #czas trwania sygnalu
fs=16000 #czestotliwosc probkowania fs>=2*fmax
f=8000 #czestotliwosc
Ts=1/fs #okres probkowania
N=round(Tc*fs,0) #liczba probek na caly sygnal
czas3=[]
tabu3=[]
for n in range(0,N,1):
    t=n*Ts
    if t>=0 and t<1.2:
        u=(-t**2+0.5)*m.sin(30*m.pi*t)*m.log2(t**2+1)
        tabu3.append(u)
        czas3.append(t)
    if t>=1.2 and t<2:
        u=(1/t)*0.8*m.sin(25*m.pi*t)-0.1*t
        tabu3.append(u)
        czas3.append(t)
    if t>=2 and t<2.4:
        u=abs(m.sin(2*m.pi*t**2))**0.8
        tabu3.append(u)
        czas3.append(t)
    if t>=2.4 and t<3.1:
        u=0.23*m.sin(20*m.pi*t)*m.sin(12*m.pi*t)
        tabu3.append(u)
        czas3.append(t)
       
DFT(tabu3)
zadanie3_koniec=time.time()
WidmoAmplitudowe()
plt.subplot(1,2,1)
WidmoDecybel()
plt.subplot(1,2,2)
WidmoDecybel()
plt.xlim(0,40)
plt.show()
CzestProzkowania(N)
zadanie3_czas=zadanie3_koniec-zadanie3_start
kolejne_czasy_DFT.append(zadanie3_czas)
print('Ca³kowity czas wykonania DFT dla funkcji u(t) w zadaniu 3 wynosci: ',zadanie3_czas)


zad3_start=time.time()
zad3=fftpack.fft(tabu3)
zad3_koniec=time.time()
zad3_czas=zad3_koniec-zad3_start
kolejne_czasy_FFT.append(zad3_czas)
print('Czas wykonania FFT dla zad 3: ',zad3_czas)
print(kolejne_czasy_FFT)
print(kolejne_czasy_DFT)