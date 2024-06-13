from scipy.interpolate import interp1d
import math as m
import numpy as np
import scipy.fftpack
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt

#s, Fs = sf.read('sin_60Hz.wav',dtype=np.int32)
#s, Fs = sf.read('sin_440Hz.wav',dtype=np.int32)
#s, Fs = sf.read('sin_8000Hz.wav',dtype=np.int32)
#s, Fs = sf.read('sin_combined.wav',dtype=np.int32)
#s, Fs = sf.read('sing_high1.wav',dtype=np.int32)
#s, Fs = sf.read('sing_low1.wav',dtype=np.int32)
s, Fs = sf.read('sing_medium1.wav',dtype=np.int32)

sd.play(s,Fs)
status = sd.wait()

def decymacja(sygnal,n):
    nowe_wartosci=sygnal[::n]
    zakres=int(len(nowe_wartosci)/(Fs/n))
    zakres_x=np.linspace(0,1,len(nowe_wartosci))
    return nowe_wartosci, zakres_x


def interpolacja(sygnal,Fs,metoda):
    x=np.arange(0,len(sygnal),1)
    if(metoda=='liniowa'):
        metode_lin=interp1d(x,sygnal)
        x1=np.arange(0,Fs,1)
        y_lin=metode_lin(x1).astype(sygnal.dtype)
        return y_lin

    if(metoda=='nieliniowa'):
        metode_nonlin=interp1d(x,sygnal, kind='cubic')
        x1=np.arange(0,Fs,1)
        y_nonlin=metode_nonlin(x1).astype(sygnal.dtype)
        return y_nonlin


def kwantyzacja(sygnal,liczba_bitow):
    nowe_m=-(2**liczba_bitow/2)
    nowe_n=(2**liczba_bitow/2)-1
    stare_m=float(np.iinfo(sygnal.dtype).min)
    stare_n=float(np.iinfo(sygnal.dtype).max)

    Za=(sygnal-stare_m)/(stare_n-stare_m)
    Zc=np.round((Za*(nowe_n-nowe_m))+nowe_m)

    odwrotne_Za=(Zc-nowe_m)/(nowe_n-nowe_m)
    odwrotne_Zc=(odwrotne_Za*(stare_n-stare_m))+stare_m

    return odwrotne_Zc


def widmo(sygnal):
    fsize=2**8
    yf = scipy.fftpack.fft(sygnal,fsize)
    return np.arange(0,Fs/2,Fs/fsize),20*np.log10( np.abs(yf[:fsize//2]))
    

bity=[4,8,16,24]
for i in bity:
    kw=kwantyzacja(s,i)
    widmo_kw_x,widmo_kw_y=widmo(kw)
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    plt.subplot(2,1,1)
    plt.title('Kwantyzacja do: '+str(i)+'bitów')
    plt.plot(kw)
    plt.subplot(2,1,2)
    plt.title('Widmo kwantyzacji')
    plt.plot(widmo_kw_x,widmo_kw_y)
    plt.savefig('kwantyzacja_'+str(i)+'bit.jpg')
    plt.show()


freq=[2000,4000, 8000, 16000, 24000, 41000, 16950] # 16950 (tylko interpolacja)
for i in freq:
    if(i!=16950):
        interwal_decymacji=int(48000/i)
        plt.subplots_adjust(wspace=0.4,hspace=0.4)
        plt.subplot(2,1,1)
        plt.title('Decymacja z krokiem: '+str(interwal_decymacji))
        dec,zakres=decymacja(s,interwal_decymacji)
        plt.plot(dec)
        plt.subplot(2,1,2)
        plt.title('Widmo decymacji')
        widmo_dec_x,widmo_dec_y=widmo(dec)
        plt.plot(widmo_dec_x,widmo_dec_y)
        plt.savefig('decymacja_krok_'+str(interwal_decymacji)+'_'+str(i)+'Hz.jpg')
        plt.show()

        plt.subplots_adjust(wspace=0.4,hspace=0.4)
        plt.subplot(2,1,1)
        plt.title('Interpolacja '+str(i)+'Hz')
        inter_lin=interpolacja(s,i,'liniowa')
        inter_nielin=interpolacja(s,i,'nieliniowa')
        plt.plot(inter_lin)
        plt.plot(inter_nielin)
        plt.legend(['Interpolacja liniowa','Interpolacja nieliniowa'],loc='upper right')
        
        plt.subplot(2,1,2)
        widmo_inter_lin_x,widmo_inter_lin_y=widmo(inter_lin)
        widmo_inter_nielin_x,widmo_inter_nielin_y=widmo(inter_nielin)
        plt.plot(widmo_inter_lin_x,widmo_inter_lin_y)
        plt.plot(widmo_inter_nielin_x,widmo_inter_nielin_y)
        plt.legend(['Widmo interpolacji liniowej','Widmo interpolacji nieliniowej'],loc='upper right')
        plt.savefig('interpolacja_'+str(i)+'Hz.jpg')
        plt.show()
    
    else:
        plt.subplots_adjust(wspace=0.4,hspace=0.4)
        plt.subplot(2,1,1)
        plt.title('Interpolacja '+str(i)+'Hz')
        inter_lin=interpolacja(s,i,'liniowa')
        inter_nielin=interpolacja(s,i,'nieliniowa')
        plt.plot(inter_lin)
        plt.plot(inter_nielin)
        plt.legend(['Interpolacja liniowa','Interpolacja nieliniowa'],loc='upper right')
        
        plt.subplot(2,1,2)
        widmo_inter_lin_x,widmo_inter_lin_y=widmo(inter_lin)
        widmo_inter_nielin_x,widmo_inter_nielin_y=widmo(inter_nielin)
        plt.plot(widmo_inter_lin_x,widmo_inter_lin_y)
        plt.plot(widmo_inter_nielin_x,widmo_inter_nielin_y)
        plt.legend(['Widmo interpolacji liniowej','Widmo interpolacji nieliniowej'],loc='upper right')
        plt.savefig('interpolacja_'+str(i)+'Hz.jpg')
        plt.show()