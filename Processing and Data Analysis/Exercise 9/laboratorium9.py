import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import math as m
import scipy as sp
from scipy import interpolate
from scipy import fftpack
import librosa

#1. Sygnał audio
#1.1
s,fs = sf.read('audio.wav',dtype='float32')

#1.2
sd.play(s,fs)
status = sd.wait()

#1.3
s=s[:,0]
sygnal=np.interp(s,(min(s),max(s)),(-1,1))
plt.plot(sygnal)
plt.title('Sygnał wejsciowy')
plt.show()


# 2. Zastosowanie okien kroczących
#2.1
def Energy(signal,dlugosc_ramki=0.01):
    global energia_norm
    energia=[]
    liczba_klatek=int(len(signal)/(dlugosc_ramki*fs))
    dlugosc_pojedynczej_ramki=int(fs*dlugosc_ramki)

    start=0
    stop=int(dlugosc_pojedynczej_ramki)    
    for i in range(0,liczba_klatek,1):
        E=0
        for n in range(int(start),int(stop),1):
            E=np.sum(signal[n]**2)
        start=stop
        stop+=dlugosc_pojedynczej_ramki
        if stop>len(signal):
            stop=len(signal)

        energia.append(E)

    tab=[]
    for i in range(0,len(energia),1):
        for j in range(0,dlugosc_pojedynczej_ramki,1):
            tab.append(energia[i])

        energia_norm=np.interp(tab,(min(tab),max(tab)),(0,1))



def Zeros(signal,dlugosc_ramki=0.01):
    global zero_norm
    zero=[]
    liczba_klatek=int(len(signal)/(dlugosc_ramki*fs))
    dlugosc_pojedynczej_ramki=int(fs*dlugosc_ramki)

    start=0
    stop=int(dlugosc_pojedynczej_ramki)    
    for i in range(0,liczba_klatek,1):
        Z=0
        for n in range(int(start),int(stop)-1,1):
            if signal[n]*signal[n+1]>=0:
                Z+=0
            elif signal[n]*signal[n+1]<0:
                Z+=1
            Z=np.sum(Z)
        start=stop
        stop+=dlugosc_pojedynczej_ramki
        if stop>len(signal):
            stop=len(signal)

        zero.append(Z)

    tab=[]
    for i in range(0,len(zero),1):
        for j in range(0,dlugosc_pojedynczej_ramki,1):
            tab.append(zero[i])

    zero_norm=np.interp(tab,(min(tab),max(tab)),(0,1))

#2.3 
#Przy małych szerokościach ramek można lepiej odczytać jak czesto sygnał przechodzi przez 0 lub jak dużo energii jest
#Im szersze okienka tym odczyt jest mniej dokładny


#2.2/2.4
Energy(s)
Zeros(s)
plt.plot(sygnal,c='g',label='Sygnal')
plt.plot(zero_norm,c='b',label='Z')
plt.plot(energia_norm,c='r',label='E')
plt.title('Ramki 10ms')
plt.legend()
plt.show()

Energy(s,0.02)
Zeros(s,0.02)
plt.plot(sygnal,c='g',label='Sygnal')
plt.plot(zero_norm,c='b',label='Z')
plt.plot(energia_norm,c='r',label='E')
plt.title('Ramki 20ms')
plt.legend()
plt.show()

Energy(s,0.05)
Zeros(s,0.05)
plt.plot(sygnal,c='g',label='Sygnal')
plt.plot(zero_norm,c='b',label='Z')
plt.plot(energia_norm,c='r',label='E')
plt.title('Ramki 50ms')
plt.legend()
plt.show()


# 3. Analiza częstotliwościowa
#3.1
s,fs = sf.read('audio.wav',dtype='float32')
s=s[:,0]
samogłoska_e=s[68000:70048]
plt.plot(sygnal,c='g',label='Sygnal')
plt.axvline(x=68000)
plt.axvline(x=70048)
plt.title('Fragment samogłoski e')
plt.show()

sd.play(samogłoska_e,fs)
status = sd.wait()

#3.2
plt.subplot(1,4,1)
plt.plot(sygnal,c='g')
plt.xlim(68000,70048)
plt.title('Signal W')
plt.xlabel('Time in ms (2048 samples)')
plt.ylabel('Signal value')

maskowania=np.hamming(len(samogłoska_e))
plt.subplot(1,4,2)
plt.plot(maskowania)
plt.title('Hamming H')
plt.xlabel('2048 samples')
plt.ylabel('Signal value')

poszerzona_samogloska=maskowania*samogłoska_e
poszerzona_samogloska=np.interp(poszerzona_samogloska,(min(poszerzona_samogloska),max(poszerzona_samogloska)),(-1,1))
plt.subplot(1,4,3)
plt.plot(poszerzona_samogloska)
plt.title('W*H')
plt.xlabel('2048 samples')
plt.ylabel('Signal value')

#3.3
widmo=np.log(np.abs(sp.fftpack.fft(poszerzona_samogloska)))
plt.subplot(1,4,4)
plt.plot(widmo,c='r')
plt.title('Amplitude spectrum')
plt.xlim(0,1000)
plt.xlabel('Frequency')
plt.ylabel('Values')
plt.show()


#4 Rozpoznawanie samoglosek ustnych
#4.1
okno=s[68000:70048]
okno=np.array(okno)
p=20

#4.2
a=librosa.lpc(okno,p)

#4.3 Librosa to pakiet Pythona do analizy muzyki i dźwięku.
#Dostarcza elementów niezbędnych do tworzenia systemów wyszukiwania informacji muzycznych.

#4.4
for i in range(0,2048-len(a),1):
    a=np.append(a,0)

#4.5
moLPC=-1*np.log(np.abs(sp.fftpack.fft(a)))
plt.plot(widmo)
plt.plot(moLPC,c='r')
plt.title('FORMANTY')
plt.show()
