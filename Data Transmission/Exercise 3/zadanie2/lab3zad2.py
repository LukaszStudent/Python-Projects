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

#zadanie 2
#funkcja nr 4 z tabeli nr 2
zadanie2_start=time.time()
Tc=1 #czas trwania sygnalu
fs=16000 #czestotliwosc probkowania fs>=2*fmax
f=8000 #czestotliwosc
Ts=1/fs #okres probkowania
N=round(Tc*fs,0) #liczba probek na caly sygnal
taby2=[]
tabz2=[]
tabv2=[]
czas2=[]
for n in range(0,N,1):
    t=n*Ts
    x = ((m.exp(-t)*m.sin(m.pi*f*t))/(2.0001+m.cos(m.pi*t)))
    tabx.append(x)
    y=-t**2*m.cos(t/0.2)*tabx[n]
    taby2.append(y)
    z=tabx[n]*m.cos(2*m.pi*t**2+m.pi)+0.276*t**2*tabx[n]
    tabz2.append(z)
    v=m.sqrt(abs(1.77-taby2[n]+tabz2[n])*m.cos(5.2*m.pi*t)+tabx[n]+4)
    tabv2.append(v)
    czas2.append(t)
    

print('zadanie 2 y(t)')
zadanie2y_start=time.time()
DFT(taby2)
zadanie2y_koniec=time.time()
WidmoAmplitudowe()
plt.subplot(1,2,1)
WidmoDecybel()
plt.subplot(1,2,2)
WidmoDecybel()
plt.xlim(3950,4050)
plt.show()
CzestProzkowania(N)
zadanie2y_czas=zadanie2y_koniec-zadanie2y_start
kolejne_czasy_DFT.append(zadanie2y_czas)
print('Czas wykonania DFT dla funkcji y(t) w zadaniu 2 wynosci: ',zadanie2y_czas)
lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear()

print('zadanie 2 z(t)')
zadanie2z_start=time.time()
DFT(tabz2)
zadanie2z_koniec=time.time()
WidmoAmplitudowe()
plt.subplot(1,2,1)
WidmoDecybel()
plt.subplot(1,2,2)
WidmoDecybel()
plt.xlim(3985,4014)
plt.show()
CzestProzkowania(N)
zadanie2z_czas=zadanie2z_koniec=zadanie2z_start
kolejne_czasy_DFT.append(zadanie2z_czas)
print('Czas wykonania DFT dla funkcji z(t) w zadaniu 2 wynosci: ',zadanie2z_czas)
lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear()

print('zadanie 2 v(t)')
zadanie2v_start=time.time()
DFT(tabv2)
zadanie2v_koniec=time.time()
WidmoAmplitudowe()
plt.subplot(2,2,1)
WidmoDecybel()
plt.subplot(2,2,2)
WidmoDecybel()
plt.xlim(-5,20)
plt.subplot(2,2,3)
WidmoDecybel()
plt.xlim(3985,4015)
plt.subplot(2,2,4)
WidmoDecybel()
plt.xlim(7980,8000)
plt.show()
CzestProzkowania(N)
zadanie2v_czas=zadanie2v_koniec=zadanie2v_start
kolejne_czasy_DFT.append(zadanie2v_czas)
print('Czas wykonania DFT dla funkcji v(t) w zadaniu 2 wynosci: ',zadanie2v_czas)

zadanie2_koniec=zadanie2y_czas+zadanie2z_czas+zadanie2v_czas
zadanie2_czas=zadanie2_koniec-zadanie2_start
print('Ca�kowity czas wykonania DFT dla zadania 2 wynosci: ',zadanie2_czas)

zad2y_start=time.time()
zad2y=fftpack.fft(taby2)
zad2y_koniec=time.time()
zad2y_czas=zad2y_koniec-zad2y_start
kolejne_czasy_FFT.append(zad2y_czas)
print('Czas wykonania FFT dla zad 2 dla funckji y(t): ',zad2y_czas)


zad2z_start=time.time()
zad2z=fftpack.fft(tabz2)
zad2z_koniec=time.time()
zad2z_czas=zad2z_koniec-zad2z_start
kolejne_czasy_FFT.append(zad2z_czas)
print('Czas wykonania FFT dla zad 1: ',zad2z_czas)


zad2v_start=time.time()
zad2v=fftpack.fft(tabv2)
zad2v_koniec=time.time()
zad2v_czas=zad2v_koniec-zad2v_start
kolejne_czasy_FFT.append(zad2v_czas)
print('Czas wykonania FFT dla zad 1: ',zad2v_czas)
print(kolejne_czasy_FFT)
print(kolejne_czasy_DFT)