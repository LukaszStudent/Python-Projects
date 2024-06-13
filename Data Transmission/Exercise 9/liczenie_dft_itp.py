import math as m
import matplotlib.pyplot as plt
import numpy as np

fs=1000

lrzeczywista,lurojona,tab,czas=[],[],[],[]
def DFT(tablica):
    for k in range(0,len(tablica),1):
        sumarzecz,sumaurojona=0,0
        for n in range(0,len(tablica),1):
            rzeczywista=tablica[n]*m.cos((-2*m.pi*k*n)/len(tablica))
            urojona=tablica[n]*m.sin((-2*m.pi*k*n)/len(tablica))
            sumarzecz+=rzeczywista
            sumaurojona-=urojona
        lrzeczywista.append(sumarzecz)
        lurojona.append(sumaurojona)


lmodul,ltg,lalfa,index=[],[],[],[]
def WidmoAmplitudowe(tablica):
    for k in range(0,int(len(tablica)/2),1):
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
def WidmoDecybel(tablica):
    for k in range(0,int(len(tablica)/2),1):
        modul=10*m.log10(lmodul[k])
        modul_prim.append(modul)

    plt.axhline(y=max(modul_prim),color='r',label='0dB (poziom odniesienia)')
    plt.axhline(y=max(modul_prim)-3,color='g',label='3dB')
    plt.axhline(y=max(modul_prim)-6,color='y',label='6dB')
    plt.axhline(y=max(modul_prim)-12,color='k',label='12dB')

    plt.legend()
    plt.plot(modul_prim)
    plt.ylabel('A [dB]')
    plt.xlabel('f [Hz]')
    #plt.show()


lfk=[]
def CzestProzkowania(tablica):
    for k in range(0,int(len(tablica)/2),1):
        fk=k*(fs/len(tablica))
        lfk.append(fk)


def Szczerokosc():
    #print('max',max(modul_prim))
    #print(modul_prim[121])
    #print(modul_prim[modul_prim.index(max(modul_prim))+1])
    wieksze_od_3db=[i for i in modul_prim if i>max(modul_prim)-3]
    if len(wieksze_od_3db)==1:
        #print('jeden pkt przeciecia 3db')
        #print(((max(modul_prim)+abs(modul_prim[121]))-((max(modul_prim)+abs(modul_prim[121]))-6))/(max(modul_prim)+abs(modul_prim[121])))
        kraniec_prawo_od_srodka_3db_1=((max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))+1]))-((max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))+1]))-3))/(max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))+1]))
        kraniec_lewo_od_srodka_3db_1=((max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))-1]))-((max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))-1]))-3))/(max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))-1]))
        koniec_prawy3db_1=modul_prim.index(max(modul_prim))+kraniec_prawo_od_srodka_3db_1
        koniec_lewy3db_1=modul_prim.index(max(modul_prim))-kraniec_lewo_od_srodka_3db_1
        #print('Poczatek 3db',koniec_lewy3db_1)
        #print('Koniec 3db',koniec_prawy3db_1)
        szerokosc3db_1=koniec_prawy3db_1-koniec_lewy3db_1
        print('Szerokosc 3db',szerokosc3db_1)
    else:
        #print('przecina w kilku miejscach 3db')
        poczatek_listy3db=wieksze_od_3db[0]
        koniec_listy3db=wieksze_od_3db[-1]
        kraniec_prawo_od_srodka_3db_M=((koniec_listy3db+abs(modul_prim[modul_prim.index(koniec_listy3db)+1]))-(koniec_listy3db+abs(modul_prim[modul_prim.index(koniec_listy3db)+1])-3))/(koniec_listy3db+abs(modul_prim[modul_prim.index(koniec_listy3db)+1]))
        kraniec_lewo_od_srodka_3db_M=((poczatek_listy3db+abs(modul_prim[modul_prim.index(poczatek_listy3db)-1]))-(poczatek_listy3db+abs(modul_prim[modul_prim.index(poczatek_listy3db)-1])-3))/(poczatek_listy3db+abs(modul_prim[modul_prim.index(poczatek_listy3db)-1]))
        koniec_prawy3db_M=modul_prim.index(koniec_listy3db)+kraniec_prawo_od_srodka_3db_M
        koniec_lewy3db_M=modul_prim.index(poczatek_listy3db)-kraniec_lewo_od_srodka_3db_M
        #print('Poczatek 3db',koniec_lewy3db_M)
        #print('Koniec 3db',koniec_prawy3db_M)
        szerokosc3db_M=koniec_prawy3db_M-koniec_lewy3db_M
        print('Szerokosc 3db',szerokosc3db_M)



    wieksze_od_6db=[i for i in modul_prim if i>max(modul_prim)-6]
    if len(wieksze_od_6db)==1:
        #print('jeden pkt przeciecia 6db')
        #print(((max(modul_prim)+abs(modul_prim[121]))-((max(modul_prim)+abs(modul_prim[121]))-6))/(max(modul_prim)+abs(modul_prim[121])))
        kraniec_prawo_od_srodka_6db_1=((max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))+1]))-((max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))+1]))-6))/(max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))+1]))
        kraniec_lewo_od_srodka_6db_1=((max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))-1]))-((max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))-1]))-6))/(max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))-1]))
        koniec_prawy6db_1=modul_prim.index(max(modul_prim))+kraniec_prawo_od_srodka_6db_1
        koniec_lewy6db_1=modul_prim.index(max(modul_prim))-kraniec_lewo_od_srodka_6db_1
        #print('Poczatek 6db',koniec_lewy6db_1)
        #print('Koniec 6db',koniec_prawy6db_1)
        szerokosc6db_1=koniec_prawy6db_1-koniec_lewy6db_1
        print('Szerokosc 6db',szerokosc6db_1)
    else:
        #print('przecina w kilku miejscach 6db')
        poczatek_listy6db=wieksze_od_6db[0]
        koniec_listy6db=wieksze_od_6db[-1]
        kraniec_prawo_od_srodka_6db_M=((koniec_listy6db+abs(modul_prim[modul_prim.index(koniec_listy6db)+1]))-(koniec_listy6db+abs(modul_prim[modul_prim.index(koniec_listy6db)+1])-6))/(koniec_listy6db+abs(modul_prim[modul_prim.index(koniec_listy6db)+1]))
        kraniec_lewo_od_srodka_6db_M=((poczatek_listy6db+abs(modul_prim[modul_prim.index(poczatek_listy6db)-1]))-(poczatek_listy6db+abs(modul_prim[modul_prim.index(poczatek_listy6db)-1])-6))/(poczatek_listy6db+abs(modul_prim[modul_prim.index(poczatek_listy6db)-1]))
        koniec_prawy6db_M=modul_prim.index(koniec_listy6db)+kraniec_prawo_od_srodka_6db_M
        koniec_lewy6db_M=modul_prim.index(poczatek_listy6db)-kraniec_lewo_od_srodka_6db_M
        #print('Poczatek 6db',koniec_lewy6db_M)
        #print('Koniec 6db',koniec_prawy6db_M)
        szerokosc6db_M=koniec_prawy6db_M-koniec_lewy6db_M
        print('Szerokosc 6db',szerokosc6db_M)




    wieksze_od_12db=[i for i in modul_prim if i>max(modul_prim)-12]
    if len(wieksze_od_12db)==1:
        #print('jeden pkt przeciecia 12db')
        #print(((max(modul_prim)+abs(modul_prim[121]))-((max(modul_prim)+abs(modul_prim[121]))-6))/(max(modul_prim)+abs(modul_prim[121])))
        kraniec_prawo_od_srodka_12db_1=((max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))+1]))-((max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))+1]))-12))/(max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))+1]))
        kraniec_lewo_od_srodka_12db_1=((max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))-1]))-((max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))-1]))-12))/(max(modul_prim)+abs(modul_prim[modul_prim.index(max(modul_prim))-1]))
        koniec_prawy12db_1=modul_prim.index(max(modul_prim))+kraniec_prawo_od_srodka_12db_1
        koniec_lewy12db_1=modul_prim.index(max(modul_prim))-kraniec_lewo_od_srodka_12db_1
        #print('Poczatek 12db',koniec_lewy12db_1)
        #print('Koniec 12db',koniec_prawy12db_1)
        szerokosc12db_1=koniec_prawy12db_1-koniec_lewy12db_1
        print('Szerokosc 12db',szerokosc12db_1)
    else:
        #print('przecina w kilku miejscach 3db')
        poczatek_listy12db=wieksze_od_12db[0]
        koniec_listy12db=wieksze_od_12db[-1]
        kraniec_prawo_od_srodka_12db_M=((koniec_listy12db+abs(modul_prim[modul_prim.index(koniec_listy12db)+1]))-(koniec_listy12db+abs(modul_prim[modul_prim.index(koniec_listy12db)+1])-12))/(koniec_listy12db+abs(modul_prim[modul_prim.index(koniec_listy12db)+1]))
        kraniec_lewo_od_srodka_12db_M=((poczatek_listy12db+abs(modul_prim[modul_prim.index(poczatek_listy12db)-1]))-(poczatek_listy12db+abs(modul_prim[modul_prim.index(poczatek_listy12db)-1])-12))/(poczatek_listy12db+abs(modul_prim[modul_prim.index(poczatek_listy12db)-1]))
        koniec_prawy12db_M=modul_prim.index(koniec_listy12db)+kraniec_prawo_od_srodka_12db_M
        koniec_lewy12db_M=modul_prim.index(poczatek_listy12db)-kraniec_lewo_od_srodka_12db_M
        #print('Poczatek 12db',koniec_lewy12db_M)
        #print('Koniec 12db',koniec_prawy12db_M)
        szerokosc12db_M=koniec_prawy12db_M-koniec_lewy12db_M
        print('Szerokosc 12db',szerokosc12db_M)
        print('\n')


lrzeczywista.clear(),lurojona.clear(),lmodul.clear(),ltg.clear(),lalfa.clear(),index.clear(),modul_prim.clear(),lfk.clear()