import math as m
import matplotlib.pyplot as plt
import numpy as np
import liczenie_dft_itp as pomocniczy
from liczenie_dft_itp import *

def str2bin(string):
    global lista,b
    lista,b=[],[]
    for i in string:
        x=ord(i)
        lista.clear()
        for j in range(0,8,1):
            if x==0:
                b+=lista[::-1]
                break
            else:
                if x%2==0:
                    lista.append(0)
                    x=m.floor(x/2)
                else:
                    lista.append(1)
                    x=m.floor(x/2)
    print(b)
    #plt.title('napis')
    #plt.plot(b)
    #plt.show()


#plt.subplot(4,1,1)
#str2bin('ZUT')


#Tc=1 #czas trwania sygnalu
#B=len(b) #liczba bitow
#Tb=Tc/B #czas trwania jednego bitu

#W=2
#fs=100000 #czestotliwosc probkowania fs>=2*fmax
#f=230 #czestotliwosc
#Ts=1/fs #okres probkowania
#N=round(Tc*fs,0) #liczba probek na caly sygnal

#fm=5 #liczba okresow w sygnale informacyjnym oraz odleglosc miedzy kolejnymi wiekszymi wychyleniami w widmie
#fn=W*(1/Tb) #liczba okresow w sygnalach zmodulowanych oraz miejsce najwiekszego wychylenia w widmie 
#liczba_probek_na_jeden_sygnal=int(Tb*fs)

#fn1=(W+1)/Tb
#fn2=(W+2)/Tb

class Sygnal_ASK_PSK_FSK():
    def __init__(self,sygnal,A1,A2,Tc,fs,f,fm,W):
        self.A1=A1
        self.A2=A2
        self.sygnal=sygnal
        self.Tc=Tc
        self.B=len(self.sygnal)
        self.Tb=self.Tc/self.B
        self.fs=fs
        self.f=f
        self.fm=fm
        self.Ts=1/self.fs
        self.N=round(self.Tc*self.fs,0)
        
        self.W=W
        self.fn=self.W*(1/self.Tb)
        self.liczba_probek_na_jeden_sygnal=int(self.Tb*self.fs)
        self.fn1=(self.W+1)/self.Tb
        self.fn2=(self.W+2)/self.Tb

    def ASK(self):
        global lask
        czasASK=[]
        lask=[]
        start=0
        stop=self.liczba_probek_na_jeden_sygnal

        for n in range(0,self.B,1):
            for i in range(start,stop,1):
                t=i*self.Ts
                czasASK.append(t)
                if self.sygnal[n]==0:
                    za_b0=self.A1*m.sin(2*m.pi*self.fn*czasASK[i])
                    lask.append(za_b0)
                else:
                    za_b1=self.A2*m.sin(2*m.pi*self.fn*czasASK[i])
                    lask.append(za_b1)
            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)
        print('ask')
       
        plt.title('ASK')
        plt.plot(czasASK,lask)
        #plt.show()


    def PSK(self):
        global lpsk
        czasPSK=[]
        lpsk=[]
        start=0
        stop=self.liczba_probek_na_jeden_sygnal

        for n in range(0,self.B,1):
            for i in range(start,stop,1):
                t=i*self.Ts
                czasPSK.append(t)
                if self.sygnal[n]==0:
                    zp_b0=m.sin(2*m.pi*self.fn*t)
                    lpsk.append(zp_b0)
                else:
                    zp_b1=zp_b0=m.sin(2*m.pi*self.fn*t+m.pi)
                    lpsk.append(zp_b1)
            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        print('psk')
       
        plt.title('PSK')
        plt.plot(czasPSK,lpsk)
        #plt.show()



    def FSK(self):
        global lfsk
        czasFSK=[]
        lfsk=[]
        start=0
        stop=self.liczba_probek_na_jeden_sygnal

        for n in range(0,self.B,1):
            for i in range(start,stop,1):
                t=i*self.Ts
                czasFSK.append(t)
                if self.sygnal[n]==0:
                    zf_b0=m.sin(2*m.pi*self.fn1*t)
                    lfsk.append(zf_b0)
                else:
                    zf_b1=m.sin(2*m.pi*self.fn2*t)
                    lfsk.append(zf_b1)
            start=stop-1
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        print('fsk')
       
        plt.title('FSK')
        plt.plot(czasFSK,lfsk)
        #plt.show()


#plt.subplot(3,1,1)
#ASK(12,33,b)
#plt.subplot(3,1,2)
#PSK(b)
#plt.subplot(3,1,3)
#FSK(b)
#plt.show()


#plt.subplot(3,1,1)
#pomocniczy.DFT(lask)
#pomocniczy.WidmoAmplitudowe(lask)
#pomocniczy.WidmoDecybel(lask)
#pomocniczy.CzestProzkowania(lask)
#print('Szerokosc pasm dla ASK')
#pomocniczy.Szczerokosc()
#pomocniczy.lrzeczywista.clear(),pomocniczy.lurojona.clear(),pomocniczy.lmodul.clear(),pomocniczy.ltg.clear(),pomocniczy.lalfa.clear(),pomocniczy.index.clear(),pomocniczy.modul_prim.clear(),pomocniczy.lfk.clear()


#plt.subplot(3,1,2)
#pomocniczy.DFT(lpsk)
#pomocniczy.WidmoAmplitudowe(lpsk)
#pomocniczy.WidmoDecybel(lpsk)
#pomocniczy.CzestProzkowania(lpsk)
#print('Szerokosc pasm dla PSK')
#pomocniczy.Szczerokosc()
#pomocniczy.lrzeczywista.clear(),pomocniczy.lurojona.clear(),pomocniczy.lmodul.clear(),pomocniczy.ltg.clear(),pomocniczy.lalfa.clear(),pomocniczy.index.clear(),pomocniczy.modul_prim.clear(),pomocniczy.lfk.clear()

#plt.subplot(3,1,3)
#pomocniczy.DFT(lfsk)
#pomocniczy.WidmoAmplitudowe(lfsk)
#pomocniczy.WidmoDecybel(lfsk)
#pomocniczy.CzestProzkowania(lfsk)
#print('Szerokosc pasm dla FSK')
#pomocniczy.Szczerokosc()
#pomocniczy.lrzeczywista.clear(),pomocniczy.lurojona.clear(),pomocniczy.lmodul.clear(),pomocniczy.ltg.clear(),pomocniczy.lalfa.clear(),pomocniczy.index.clear(),pomocniczy.modul_prim.clear(),pomocniczy.lfk.clear()
#plt.show()

#lask.clear(),lpsk.clear(),lfsk.clear()


#zadanie 2 z ograniczeniem do 10bitow
#b_10=b[0:10]
#B=len(b_10)
#print('\n Zadanie 2')
#print(b_10)
#plt.subplot(3,1,1)
#ASK(12,33,b_10)
#plt.subplot(3,1,2)
#PSK(b_10)
#plt.subplot(3,1,3)
#FSK(b_10)
#plt.show()


#plt.subplot(3,1,1)
#pomocniczy.DFT(lask)
#pomocniczy.WidmoAmplitudowe(lask)
#pomocniczy.WidmoDecybel(lask)
#pomocniczy.CzestProzkowania(lask)
#print('Szerokosc pasm dla ASK')
#pomocniczy.Szczerokosc()
#pomocniczy.lrzeczywista.clear(),pomocniczy.lurojona.clear(),pomocniczy.lmodul.clear(),pomocniczy.ltg.clear(),pomocniczy.lalfa.clear(),pomocniczy.index.clear(),pomocniczy.modul_prim.clear(),pomocniczy.lfk.clear()


#plt.subplot(3,1,2)
#pomocniczy.DFT(lpsk)
#pomocniczy.WidmoAmplitudowe(lpsk)
#pomocniczy.WidmoDecybel(lpsk)
#pomocniczy.CzestProzkowania(lpsk)
#print('Szerokosc pasm dla PSK')
#pomocniczy.Szczerokosc()
#pomocniczy.lrzeczywista.clear(),pomocniczy.lurojona.clear(),pomocniczy.lmodul.clear(),pomocniczy.ltg.clear(),pomocniczy.lalfa.clear(),pomocniczy.index.clear(),pomocniczy.modul_prim.clear(),pomocniczy.lfk.clear()

#plt.subplot(3,1,3)
#pomocniczy.DFT(lfsk)
#pomocniczy.WidmoAmplitudowe(lfsk)
#pomocniczy.WidmoDecybel(lfsk)
#pomocniczy.CzestProzkowania(lfsk)
#print('Szerokosc pasm dla FSK')
#pomocniczy.Szczerokosc()
#pomocniczy.lrzeczywista.clear(),pomocniczy.lurojona.clear(),pomocniczy.lmodul.clear(),pomocniczy.ltg.clear(),pomocniczy.lalfa.clear(),pomocniczy.index.clear(),pomocniczy.modul_prim.clear(),pomocniczy.lfk.clear()
#plt.show()
