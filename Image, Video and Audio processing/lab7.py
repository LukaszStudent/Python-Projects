from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd

s, Fs = sf.read('sin_60Hz.wav')
s = s.astype(float)

# sd.play(s,Fs)
# status = sd.wait()
A=87.6

def A_law_kompresja(zakres):
    sygnal=[]
    for x in zakres:
        if(0<np.abs(x)<(1/A)):
            wartosc=np.sign(x)*(A*np.abs(x))/(1+np.log(A))
            sygnal.append(wartosc)
        if((1/A)<=np.abs(x)<=1):
            wartosc=np.sign(x)*(1+np.log(A*np.abs(x)))/(1+np.log(A))
            sygnal.append(wartosc)
    return sygnal


def A_law_dekompresja(zakres):
    sygnal=[]
    for y in zakres:
        if(np.abs(y)<1/(1+np.log(A))):
            wartosc=np.sign(y)*np.abs(y)*(1+np.log(A))/A
            sygnal.append(wartosc)
        if(1/(1+np.log(A))<=np.abs(y) and np.abs(y)<=1):
            wartosc=np.sign(y)*np.exp(np.abs(y)*(1+np.log(A))-1)/A
            sygnal.append(wartosc)
    return np.arange(len(sygnal)),sygnal


def Mu_law_kompresja(zakres):
    sygnal=[]
    for x in zakres:
        if(-1<=x and x<=1):
            wartosc=np.sign(x)*(np.log(1+255*np.abs(x))/np.log(1+255))
            sygnal.append(wartosc)
    return sygnal


def Mu_law_dekompresja(zakres):
    sygnal=[]
    for y in zakres:
        if(-1<=y and y<=1):
            wartosc=np.sign(y)*(1/255)*((1+255)**np.abs(y)-1)
            sygnal.append(wartosc)
    return np.arange(len(sygnal)),sygnal


def kwantyzacja(sygnal,liczba_bitow):
    zakres=np.linspace(-1,1,2**liczba_bitow)
    nowy_sygnal=[]
    for i in range(len(sygnal)):
        #wartosc=np.round(sygnalFit(sygnal[i],zakres),2)
        wartosc=sygnalFit(sygnal[i],zakres)
        nowy_sygnal.append(wartosc)
    return nowy_sygnal


def sygnalFit(wartosc_sygnalu,zakres):
    zakres = np.asarray(zakres)
    idx = (np.abs(zakres - wartosc_sygnalu)).argmin()
    return zakres[idx]


def kwantyzacja_DPCM(wartosc,tablica):
    zakres=np.arange(wartosc-2,wartosc+2,1)
    najblizsza=(np.abs(zakres-wartosc)).argmin()
    for i in zakres:
        if(wartosc%2!=0 and i/wartosc==1):
            return zakres[najblizsza+1]
        if(wartosc%2==0 and i/wartosc==1):
            return zakres[najblizsza]
        if(wartosc==0):
            return zakres[najblizsza]
        else:
            return zakres[najblizsza]

def DPCM_koder_bez_predykcji(sygnal):
    nowy_sygnal=[]
    e=0
    for i in range(len(sygnal)):
        X=sygnal[i]
        Y_prim=(X-e)
        Y=kwantyzacja_DPCM(Y_prim,sygnal)
        nowy_sygnal.append(Y)
        X_prim=Y+e
        e=X_prim
    
    return nowy_sygnal

def DPCM_dekoder_bez_predykcji(sygnal_zakodowany):
    nowy_sygnal=[]
    X=0
    for i in range(len(sygnal_zakodowany)):
        Y=sygnal_zakodowany[i]
        nowy_sygnal.append(Y+X)
        X=Y+X

    return np.arange(len(nowy_sygnal)),nowy_sygnal


# X=[15,16,20,14,5,10,15,13,11,7,10,11,20,1,23]
# print('Oryginalny sygnal:\n',X)
# dpcm_koder=DPCM_koder_bez_predykcji(X)
# print('Kompresja DPCM:\n',dpcm_koder)
# dpcm_dekoder=DPCM_dekoder_bez_predykcji(dpcm_koder)
# print('Dekompresja DPCM:\n',dpcm_dekoder)


x=np.linspace(-1,1,10000)
y=0.9*np.sin(np.pi*x*4)

sygnal_oryginalny_kwantyzacja_8_bit=kwantyzacja(x,8)
sygnal_kompresja_A_law=A_law_kompresja(x)
sygnal_kompresja_A_law_8_bit=kwantyzacja(sygnal_kompresja_A_law,8)
x_Alaw,sygnal_dekompresja_A_law=A_law_dekompresja(sygnal_kompresja_A_law_8_bit)

sygnal_kompresja_Mu_law=Mu_law_kompresja(x)
sygnal_kompresja_Mu_law_8_bit=kwantyzacja(sygnal_kompresja_Mu_law,8)
x_Mulaw,sygnal_dekompresja_Mu_law=Mu_law_dekompresja(sygnal_kompresja_Mu_law_8_bit)
plt.subplot(1,2,1)
plt.title('Krzywa kompresji')
plt.plot(x,sygnal_kompresja_A_law_8_bit,c='b')
plt.plot(x,sygnal_kompresja_A_law,c='orange')
plt.plot(x,sygnal_kompresja_Mu_law_8_bit,c='g')
plt.plot(x,sygnal_kompresja_Mu_law,c='r')
plt.xlabel('Wartości sygnału wejściowego')
plt.ylabel('Wartości sygnału wyjściowego')
plt.legend(['Sygnał po kompresji a-law po kwantyzacji do 8 bitów','Sygnał po kompresji a-law bez kwantyzacji','Sygnał po kompresji mu-law po kwantyzacji do 8 bitów','Sygnał po kompresji mu-law bez kwantyzacji'],loc='upper left')

plt.subplot(1,2,2)
plt.title('Krzywa po dekompresji')
plt.plot(x,x,c='b')
plt.plot(x,sygnal_dekompresja_A_law,c='orange')
plt.plot(x,sygnal_dekompresja_Mu_law,c='g')
plt.plot(x,sygnal_oryginalny_kwantyzacja_8_bit,c='r')
plt.legend(['Sygnał oryginalny','Sygnał po dekompresji z a-law (kwantyzacja 8-bitów)','Sygnał po dekompresji z mu-law (kwantyzacja 8-bitów)','Sygnał oryginalny po kwantyzacji do 8 bitów'],loc='upper left')
plt.show()


plt.subplot(1,2,1)
plt.title('Krzywa kompresji')
plt.plot(x,sygnal_kompresja_A_law_8_bit,c='b')
plt.plot(x,sygnal_kompresja_A_law,c='orange')
plt.plot(x,sygnal_kompresja_Mu_law_8_bit,c='g')
plt.plot(x,sygnal_kompresja_Mu_law,c='r')
plt.xlim(-0.9,-0.8)
plt.ylim(-0.987,-0.955)
plt.legend(['Sygnał po kompresji a-law po kwantyzacji do 8 bitów','Sygnał po kompresji a-law bez kwantyzacji','Sygnał po kompresji mu-law po kwantyzacji do 8 bitów','Sygnał po kompresji mu-law bez kwantyzacji'],loc='upper left')

plt.subplot(1,2,2)
plt.title('Krzywa po dekompresji')
plt.plot(x,x,c='b')
plt.plot(x,sygnal_dekompresja_A_law,c='orange')
plt.plot(x,sygnal_dekompresja_Mu_law,c='g')
plt.plot(x,sygnal_oryginalny_kwantyzacja_8_bit,c='r')
plt.xlim(-0.9,-0.8)
plt.ylim(-0.92,-0.78)
plt.legend(['Sygnał oryginalny','Sygnał po dekompresji z a-law (kwantyzacja 8-bitów)','Sygnał po dekompresji z mu-law (kwantyzacja 8-bitów)','Sygnał oryginalny po kwantyzacji do 8 bitów'],loc='upper left')
plt.show()


plt.subplot(1,2,1)
plt.title('Krzywa kompresji')
plt.plot(x,sygnal_kompresja_A_law_8_bit,c='b')
plt.plot(x,sygnal_kompresja_A_law,c='orange')
plt.plot(x,sygnal_kompresja_Mu_law_8_bit,c='g')
plt.plot(x,sygnal_kompresja_Mu_law,c='r')
plt.xlim(-0.01,0.01)
plt.ylim(-0.2,0.2)
plt.legend(['Sygnał po kompresji a-law po kwantyzacji do 8 bitów','Sygnał po kompresji a-law bez kwantyzacji','Sygnał po kompresji mu-law po kwantyzacji do 8 bitów','Sygnał po kompresji mu-law bez kwantyzacji'],loc='upper left')
plt.subplot(1,2,2)
plt.title('Krzywa po dekompresji')
plt.plot(x,x,c='b')
plt.plot(x,sygnal_dekompresja_A_law,c='orange')
plt.plot(x,sygnal_dekompresja_Mu_law,c='g')
plt.plot(x,sygnal_oryginalny_kwantyzacja_8_bit,c='r')
plt.xlim(-0.01,0.01)
plt.ylim(-0.012,0.012)
plt.legend(['Sygnał oryginalny','Sygnał po dekompresji z a-law (kwantyzacja 8-bitów)','Sygnał po dekompresji z mu-law (kwantyzacja 8-bitów)','Sygnał oryginalny po kwantyzacji do 8 bitów'],loc='upper left')
plt.show()



y=s
x=np.arange(0,48000,1)
stopien_kwantyzacji=3
sygnal_kompresja_A_law=A_law_kompresja(y)
sygnal_kompresja_A_law_8_bit=kwantyzacja(sygnal_kompresja_A_law,stopien_kwantyzacji)
x_Alaw,sygnal_dekompresja_A_law=A_law_dekompresja(sygnal_kompresja_A_law_8_bit)

sygnal_kompresja_Mu_law=Mu_law_kompresja(y)
sygnal_kompresja_Mu_law_8_bit=kwantyzacja(sygnal_kompresja_Mu_law,stopien_kwantyzacji)
x_Mulaw,sygnal_dekompresja_Mu_law=Mu_law_dekompresja(sygnal_kompresja_Mu_law_8_bit)

sygnal_kompresja_DPCM=DPCM_koder_bez_predykcji(y)
x_DPCM,sygnal_dekompresja_DPCM=DPCM_dekoder_bez_predykcji(sygnal_kompresja_DPCM)

plt.subplots_adjust(wspace=0.4,hspace=0.4)
plt.suptitle('Przykład A\nPoziom kwantyzacji: '+str(stopien_kwantyzacji)+'\n')
plt.subplot(4,1,1)
plt.title('Sygnał oryginalny')
plt.plot(x,y)
plt.subplot(4,1,2)
plt.title('Kompresja A-law')
plt.plot(x_Alaw,sygnal_dekompresja_A_law)
plt.subplot(4,1,3)
plt.title('Kompresja Mu-law')
plt.plot(x_Mulaw,sygnal_dekompresja_Mu_law)
plt.subplot(4,1,4)
plt.title('Kompresja DPCM')
plt.plot(x_DPCM,sygnal_dekompresja_DPCM)
plt.show()


plt.suptitle('Przykład B\nPoziom kwantyzacji: '+str(stopien_kwantyzacji)+'\n')
plt.plot(x,y,c='b')
plt.plot(x_Alaw,sygnal_dekompresja_A_law,c='orange')
plt.plot(x_Mulaw,sygnal_dekompresja_Mu_law,c='g')
plt.plot(x_DPCM,sygnal_dekompresja_DPCM,c='r')
plt.legend(['Sygnał oryginalny','Sygnał po dekompresji z a-law','Sygnał po dekompresji z mu-law','Sygnał po dekompresji z DPCM'],loc='upper left')
plt.show()