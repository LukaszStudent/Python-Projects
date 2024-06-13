import math as m
import matplotlib.pyplot as plt
from scipy import fftpack
import time

kolejne_czasy_DFT=[]
kolejne_czasy_FFT=[]

Tc=1 #czas trwania sygnalu
fs=900 #czestotliwosc probkowania fs>=2*fmax
f=230 #czestotliwosc
Ts=1/fs #okres probkowania
N=round(Tc*fs,0) #liczba probek na caly sygnal
lrzeczywista,lurojona,tab,czas=[],[],[],[]

#funkcja z zajec
for n in range(0,N,1):
    t=n*Ts
    x=m.sin(2*m.pi*f*t)
    tab.append(x)
    czas.append(t)

plt.subplot(2,1,1)
plt.plot(czas,tab)
plt.xlim(0,0.5)
plt.subplot(2,1,2)
plt.plot(czas,tab)
plt.show()


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
    plt.plot(lfk,lmodul)
    plt.show()

przyklad_start=time.time()
DFT(tab)
WidmoAmplitudowe()
WidmoDecybel()
CzestProzkowania(N)
przyklad_koniec=time.time()
przyklad_czas=przyklad_koniec-przyklad_start
print('Czas wykonania DFT dla przykladu z zajec wynosi: ',przyklad_czas)
lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear()

##zadanie 1
#zadanie1_start=time.time()
#print('zadanie 1')
#Tc=1 #czas trwania sygnalu
#fs=16000 #czestotliwosc probkowania fs>=2*fmax #zmienione z 20000
#f=8000 #czestotliwosc #zmienione z 10000
#Ts=1/fs #okres probkowania
#N=round(Tc*fs,0) #liczba probek na caly sygnal
#tabx=[]
#for n in range(0,N,1):
#    t=n*Ts
#    x=((m.exp(-t)*m.sin(m.pi*f*t))/(2.0001+m.cos(m.pi*t)))
#    tabx.append(x)
#    czas.append(t)

#DFT(tabx)
#zadanie1_koniec=time.time()
#WidmoAmplitudowe()
#plt.subplot(1,2,1)
#WidmoDecybel()
#plt.subplot(1,2,2)
#WidmoDecybel()
#plt.xlim(3950,4050)
#plt.show()
#CzestProzkowania(N)

#zadanie1_czas=zadanie1_koniec-zadanie1_start
#print('Czas wykonania DFT dla zadanie 1 wynosci: ',zadanie1_czas)
#kolejne_czasy_DFT.append(zadanie1_czas)
#lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear()

##zadanie 2
##funkcja nr 4 z tabeli nr 2
#zadanie2_start=time.time()
#Tc=1 #czas trwania sygnalu
#fs=16000 #czestotliwosc probkowania fs>=2*fmax
#f=8000 #czestotliwosc
#Ts=1/fs #okres probkowania
#N=round(Tc*fs,0) #liczba probek na caly sygnal
#taby2=[]
#tabz2=[]
#tabv2=[]
#czas2=[]
#for n in range(0,N,1):
#    t=n*Ts
#    x = ((m.exp(-t)*m.sin(m.pi*f*t))/(2.0001+m.cos(m.pi*t)))
#    tabx.append(x)
#    y=-t**2*m.cos(t/0.2)*tabx[n]
#    taby2.append(y)
#    z=tabx[n]*m.cos(2*m.pi*t**2+m.pi)+0.276*t**2*tabx[n]
#    tabz2.append(z)
#    v=m.sqrt(abs(1.77-taby2[n]+tabz2[n])*m.cos(5.2*m.pi*t)+tabx[n]+4)
#    tabv2.append(v)
#    czas2.append(t)
    

#print('zadanie 2 y(t)')
#zadanie2y_start=time.time()
#DFT(taby2)
#zadanie2y_koniec=time.time()
#WidmoAmplitudowe()
#plt.subplot(1,2,1)
#WidmoDecybel()
#plt.subplot(1,2,2)
#WidmoDecybel()
#plt.xlim(3950,4050)
#plt.show()
#CzestProzkowania(N)
#zadanie2y_czas=zadanie2y_koniec-zadanie2y_start
#kolejne_czasy_DFT.append(zadanie2y_czas)
#print('Czas wykonania DFT dla funkcji y(t) w zadaniu 2 wynosci: ',zadanie2y_czas)
#lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear()

#print('zadanie 2 z(t)')
#zadanie2z_start=time.time()
#DFT(tabz2)
#zadanie2z_koniec=time.time()
#WidmoAmplitudowe()
#plt.subplot(1,2,1)
#WidmoDecybel()
#plt.subplot(1,2,2)
#WidmoDecybel()
#plt.xlim(3985,4014)
#plt.show()
#CzestProzkowania(N)
#zadanie2z_czas=zadanie2z_koniec=zadanie2z_start
#kolejne_czasy_DFT.append(zadanie2z_czas)
#print('Czas wykonania DFT dla funkcji z(t) w zadaniu 2 wynosci: ',zadanie2z_czas)
#lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear()

#print('zadanie 2 v(t)')
#zadanie2v_start=time.time()
#DFT(tabv2)
#zadanie2v_koniec=time.time()
#WidmoAmplitudowe()
#plt.subplot(2,2,1)
#WidmoDecybel()
#plt.subplot(2,2,2)
#WidmoDecybel()
#plt.xlim(-5,20)
#plt.subplot(2,2,3)
#WidmoDecybel()
#plt.xlim(3985,4015)
#plt.subplot(2,2,4)
#WidmoDecybel()
#plt.xlim(7980,8000)
#plt.show()
#CzestProzkowania(N)
#zadanie2v_czas=zadanie2v_koniec=zadanie2v_start
#kolejne_czasy_DFT.append(zadanie2v_czas)
#print('Czas wykonania DFT dla funkcji v(t) w zadaniu 2 wynosci: ',zadanie2v_czas)

#zadanie2_koniec=zadanie2y_czas+zadanie2z_czas+zadanie2v_czas
#zadanie2_czas=zadanie2_koniec-zadanie2_start
#print('Całkowity czas wykonania DFT dla zadania 2 wynosci: ',zadanie2_czas)
#lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear()

#zadanie 3
#zadanie3_start=time.time()
#Tc=1 #czas trwania sygnalu
#fs=16000 #czestotliwosc probkowania fs>=2*fmax
#f=8000 #czestotliwosc
#Ts=1/fs #okres probkowania
#N=round(Tc*fs,0) #liczba probek na caly sygnal
#czas3=[]
#tabu3=[]
#for n in range(0,N,1):
#    t=n*Ts
#    if t>=0 and t<1.2:
#        u=(-t**2+0.5)*m.sin(30*m.pi*t)*m.log2(t**2+1)
#        tabu3.append(u)
#        czas3.append(t)
#    if t>=1.2 and t<2:
#        u=(1/t)*0.8*m.sin(25*m.pi*t)-0.1*t
#        tabu3.append(u)
#        czas3.append(t)
#    if t>=2 and t<2.4:
#        u=abs(m.sin(2*m.pi*t**2))**0.8
#        tabu3.append(u)
#        czas3.append(t)
#    if t>=2.4 and t<3.1:
#        u=0.23*m.sin(20*m.pi*t)*m.sin(12*m.pi*t)
#        tabu3.append(u)
#        czas3.append(t)
       
#DFT(tabu3)
#zadanie3_koniec=time.time()
#WidmoAmplitudowe()
#plt.subplot(1,2,1)
#WidmoDecybel()
#plt.subplot(1,2,2)
#WidmoDecybel()
#plt.xlim(0,40)
#plt.show()
#CzestProzkowania(N)
#zadanie3_czas=zadanie3_koniec-zadanie3_start
#kolejne_czasy_DFT.append(zadanie3_czas)
#print('Całkowity czas wykonania DFT dla funkcji u(t) w zadaniu 3 wynosci: ',zadanie3_czas)
#lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear()

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
#DFT(lista1)
#zadanie4_1_koniec=time.time()
#WidmoAmplitudowe()
#plt.subplot(1,2,1)
#WidmoDecybel()
#plt.subplot(1,2,2)
#WidmoDecybel()
#plt.xlim(11015,11025)
#plt.show()
#CzestProzkowania(N)
#zadanie4_1_czas=zadanie4_1_koniec-zadanie4_1_start
#kolejne_czasy_DFT.append(zadanie4_1_czas)
#print('Czas wykonania DFT dla pierwszego przedziału w zadaniu 4 wynosci: ',zadanie4_1_czas)
#lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear()

#print('zadanie 4 2')
#zadanie4_2_start=time.time()
#DFT(lista2)
#zadanie4_2_koniec=time.time()
#WidmoAmplitudowe()
#plt.subplot(2,2,1)
#WidmoDecybel()
#plt.subplot(2,2,2)
#WidmoDecybel()
#plt.xlim(3660,3690)
#plt.subplot(2,2,3)
#WidmoDecybel()
#plt.xlim(7330,7370)
#plt.subplot(2,2,4)
#WidmoDecybel()
#plt.xlim(11010,11025)
#plt.show()
#CzestProzkowania(N)
#zadanie4_2_czas=zadanie4_2_koniec-zadanie4_2_start
#kolejne_czasy_DFT.append(zadanie4_2_czas)
#print('Czas wykonania DFT dla drugiego przedziału w zadaniu 4 wynosci: ',zadanie4_2_czas)
#lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear(),czas.clear()

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
print('Czas wykonania DFT dla trzeciego przedziału w zadaniu 4 wynosci: ',zadanie4_3_czas)
lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear()

zad1_start=time.time()
zad1=fftpack.fft(tabx)
zad1_koniec=time.time()
zad1_czas=zad1_koniec-zad1_start
kolejne_czasy_FFT.append(zad1_czas)
print('Czas wykonania FFT dla zad 1: ',zad1_czas)


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


zad3_start=time.time()
zad3=fftpack.fft(tabu3)
zad3_koniec=time.time()
zad3_czas=zad3_koniec-zad3_start
kolejne_czasy_FFT.append(zad3_czas)
print('Czas wykonania FFT dla zad 3: ',zad3_czas)


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
