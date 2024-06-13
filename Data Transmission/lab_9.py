import lab_5 as SK
import lab_6 as DSK
import lab_7 as Ham
import lab_8 as H
import liczenie_dft_itp as pomocnicze
import matplotlib.pyplot as plt
import numpy as np
import math as m

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

def Szum(srednia,odchylenie_stand,sygnal,alpha):
    global lfala
    lfala=[]
    szum=np.random.normal(srednia,odchylenie_stand,len(sygnal))*alpha
    for i in range(0,len(sygnal),1):
        lfala.append(szum[i]+sygnal[i])
    plt.plot(lfala)
    plt.title('Szum')
    plt.show()

def Tlumienie(sygnal,beta):
    global ltlumienie
    ltlumienie=[]
    for i in range(0,len(sygnal),1):
        ltlumienie.append(sygnal[i]*m.exp(-beta*i))
    plt.plot(ltlumienie)
    plt.title('Tłumienie')
    plt.show()

def BER(szum,tlumienie):
    licznik=0
    for i in range(0,len(szum),1):
        if szum[i]!=tlumienie[i]:
            licznik+=1
    ber=licznik/len(szum)
    print(ber)

slowo=[1,0,1,1,1,1,1,0,1,0,0,0,0,1,1,0,1,1,0,0,0,1,0]

ilosc_calych_blokow=int(round(len(slowo)/4,0))
ilosc_dopisanych=len(slowo)-(ilosc_calych_blokow*4)
for i in range(0,4-ilosc_dopisanych,1):
    slowo.append(0)

Hamming=Ham.Hamming_7_4(slowo)
Hamming.koder()
zakodowane=Ham.zakodowane

print('1.ASK\n2.PSK\n3.FSK')
x=int(input('Podaj w jaki sposob nadac sygnal: '))
if x==1:
    Modulator=SK.Sygnal_ASK_PSK_FSK(zakodowane,11,33,1,100000,230,5,2)
    Modulator.ASK()
    plt.show()
    Tlumienie(SK.lask,10)
    Szum(0,1,SK.lask,0.5)
    BER(lfala,ltlumienie)
    DASK=DSK.Demodulator_ASK(lfala,12,1,100000,230,5,2)
    plt.show()
    DASK.x_t()
    plt.show()
    DASK.p_t()
    plt.show()
    DASK.c_t(50)
    plt.show()
    #DASK.konwersja()
    #plt.show()

elif x==2:
    Modulator=SK.Sygnal_ASK_PSK_FSK(zakodowane,33,11,1,100000,230,5,2)
    Modulator.PSK()
    Szum(0,1,SK.lpsk,0.5)
    DPSK=DSK.Demodulator_PSK(lfala,12,1,100000,230,5,2)
    DPSK.x_t()
    plt.show()
    DPSK.p_t()
    plt.show()
    zmienna=DPSK.c_t()
    plt.show()
    DSK.konwersja(zmienna)
    plt.show()
    Hamming.dekoder(zmienna)

elif x==3:
    Modulator=SK.Sygnal_ASK_PSK_FSK(zakodowane,33,11,1,100000,230,5,2)
    Modulator.FSK()
    Szum(0,1,SK.lfsk,0.5)
    DFSK=DSK.Demodulator_FSK(lfala,12,1,100000,230,5,2)
    DFSK.x1_t()
    plt.show()
    DFSK.x2_t()
    plt.show()
    DFSK.p1_t()
    plt.show()
    DFSK.p2_t()
    plt.show()
    DFSK.p3_t()
    plt.show()
    zmienna=DFSK.c_t()
    plt.show()
    DFSK.konwersja()
    plt.show()
    Hamming.dekoder(zmienna)

else:
    print('Podales zla wartosc')
