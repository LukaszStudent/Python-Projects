import numpy as np
import matplotlib.pyplot as plt 
import scipy.fftpack
import cv2

def RGB_do_YCrCb(obraz):
    nowy_obraz=cv2.cvtColor(obraz,cv2.COLOR_RGB2YCrCb).astype(int)
    return nowy_obraz

def YCrCb_do_RGB(obraz):
    nowy_obraz=cv2.cvtColor(obraz.astype(np.uint8),cv2.COLOR_YCrCb2RGB)
    return nowy_obraz

def redukcja_Chromatycznosci(obraz,metoda):
    Y=obraz[:,:,0]
    if(metoda=='444'):
        Cr=obraz[:,:,1]
        Cb=obraz[:,:,2]
    elif(metoda=='440'):
        Cr=obraz[0::2,:,1]
        Cb=obraz[0::2,:,2]
    elif(metoda=='420'):
        Cr=obraz[0::2,0::2,1]
        Cb=obraz[0::2,0::2,2]
    elif(metoda=='422'):
        Cr=obraz[:,0::2,1]
        Cb=obraz[:,0::2,2]
    elif(metoda=='411'):
        Cr=obraz[:,0::4,1]
        Cb=obraz[:,0::4,2]
    elif(metoda=='410'):
        Cr=obraz[0::2,0::4,1]
        Cb=obraz[0::2,0::4,2]
    else:
        print('Zły parametr')

    return Y, Cr, Cb

def redukcja_Chromatycznosci_do_YCrCb(Y,Cr,Cb):
    nowe_Cr=np.zeros(Y.shape)
    nowe_Cb=np.zeros(Y.shape)
    if(Y.shape==Cr.shape==Cb.shape): #4:4:4
        Y=Y
        nowe_Cb=Cb
        nowe_Cr=Cr

    elif(Cr.shape[0]==int(Y.shape[0]/2) and Cr.shape[1]==Y.shape[1]): #4:4:0
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                nowe_Cb[i][j]=Cb[int(i/2)][j]
                nowe_Cr[i][j]=Cr[int(i/2)][j]

    elif(Cr.shape[1]==int(Y.shape[1]/2) and Cr.shape[0]==int(Y.shape[0]/2)): #4:2:0
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                nowe_Cb[i][j]=Cb[int(i/2)][int(j/2)]
                nowe_Cr[i][j]=Cr[int(i/2)][int(j/2)]
    
    elif(Cr.shape[1]==int(Y.shape[1]/2) and Cr.shape[0]==Y.shape[0]): #4:2:2
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                nowe_Cb[i][j]=Cb[i][int(j/2)]
                nowe_Cr[i][j]=Cr[i][int(j/2)]

    elif(Cr.shape[1]==int(Y.shape[1]/4) and Cr.shape[0]==Y.shape[0]): #4:1:1
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                nowe_Cb[i][j]=Cb[i][int(j/4)]
                nowe_Cr[i][j]=Cr[i][int(j/4)]

    elif(Cr.shape[1]==int(Y.shape[1]/4) and Cr.shape[0]==int(Y.shape[0]/2)): #4:1:0
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                nowe_Cb[i][j]=Cb[int(i/2)][int(j/4)]
                nowe_Cr[i][j]=Cr[int(i/2)][int(j/4)]
        
    else:
        print('Cos nie smiga')

    return Y, nowe_Cr, nowe_Cb

def dct2(obraz):
    return scipy.fftpack.dct( scipy.fftpack.dct( obraz.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(obraz):
    return scipy.fftpack.idct( scipy.fftpack.idct( obraz.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')

def kwantyzacja(Y,Cr,Cb):
    QY= np.array([
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 36, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
        ])

    QC= np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        ])

    wymiary_Y=(int(Y.shape[0]/8),int(Y.shape[1]/8))
    kw_Y=np.zeros(Y.shape)
    for i in range(wymiary_Y[0]):
        for j in range(wymiary_Y[1]):
            macierz_1=Y[i*8:(i+1)*8,j*8:(j+1)*8]
            macierz_1=dct2(macierz_1)
            macierz_1=np.round(macierz_1/QY).astype(int)
            kw_Y[i*8:(i+1)*8,j*8:(j+1)*8]=macierz_1

    wymiary_Cr=(int(Cr.shape[0]/8),int(Cr.shape[1]/8))
    kw_Cr=np.zeros(Cr.shape)
    for i in range(wymiary_Cr[0]):
        for j in range(wymiary_Cr[1]):
            macierz_1=Cr[i*8:(i+1)*8,j*8:(j+1)*8]
            macierz_1=dct2(macierz_1)
            macierz_1=np.round(macierz_1/QY).astype(int)
            kw_Cr[i*8:(i+1)*8,j*8:(j+1)*8]=macierz_1

    wymiary_Cb=(int(Cb.shape[0]/8),int(Cb.shape[1]/8))
    kw_Cb=np.zeros(Cb.shape)
    for i in range(wymiary_Cb[0]):
        for j in range(wymiary_Cb[1]):
            macierz_1=Cb[i*8:(i+1)*8,j*8:(j+1)*8]
            macierz_1=dct2(macierz_1)
            macierz_1=np.round(macierz_1/QY).astype(int)
            kw_Cb[i*8:(i+1)*8,j*8:(j+1)*8]=macierz_1

    return kw_Y,kw_Cr,kw_Cb

def dekwantyzacja(Y,Cr,Cb):
    QY= np.array([
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 36, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
        ])

    QC= np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        ])

    wymiary_Y=(int(Y.shape[0]/8),int(Y.shape[1]/8))
    kw_Y=np.zeros(Y.shape)
    for i in range(wymiary_Y[0]):
        for j in range(wymiary_Y[1]):
            macierz_1=Y[i*8:(i+1)*8,j*8:(j+1)*8]
            macierz_1=macierz_1*QY
            macierz_1=idct2(macierz_1)
            kw_Y[i*8:(i+1)*8,j*8:(j+1)*8]=macierz_1

    wymiary_Cr=(int(Cr.shape[0]/8),int(Cr.shape[1]/8))
    kw_Cr=np.zeros(Cr.shape)
    for i in range(wymiary_Cr[0]):
        for j in range(wymiary_Cr[1]):
            macierz_1=Cr[i*8:(i+1)*8,j*8:(j+1)*8]
            macierz_1=macierz_1*QY
            macierz_1=idct2(macierz_1)
            kw_Cr[i*8:(i+1)*8,j*8:(j+1)*8]=macierz_1

    wymiary_Cb=(int(Cb.shape[0]/8),int(Cb.shape[1]/8))
    kw_Cb=np.zeros(Cb.shape)
    for i in range(wymiary_Cb[0]):
        for j in range(wymiary_Cb[1]):
            macierz_1=Cb[i*8:(i+1)*8,j*8:(j+1)*8]
            macierz_1=macierz_1*QY
            macierz_1=idct2(macierz_1)
            kw_Cb[i*8:(i+1)*8,j*8:(j+1)*8]=macierz_1

    return kw_Y,kw_Cr,kw_Cb

def kompresja_bez_RLE(Y,Cr,Cb,klatka):
    rozmiar_klatki=klatka.size/3
    Y=0
    Cr=(1-Cr.size/rozmiar_klatki)*100
    Cb=(1-Cb.size/rozmiar_klatki)*100
    print(Y,Cr,Cb)

    return Y, Cr, Cb

def kompresja_RLE(macierz):
    nowa_macierz=macierz.reshape(-1)
    licznik=1
    nowa_macierz_tab=[]
    for i in range(len(nowa_macierz)):
        if(i<len(nowa_macierz)-1 and nowa_macierz[i]==nowa_macierz[i+1]):
            licznik+=1
        else:
            nowa_macierz_tab.append([licznik,nowa_macierz[i]])
            licznik=1

    nowa_macierz_tab=np.array(nowa_macierz_tab).reshape(-1)
    stopien_kompresji=len(nowa_macierz)/len(nowa_macierz_tab)
    procent_kompresji=100*1/stopien_kompresji
    return nowa_macierz_tab,macierz.shape,stopien_kompresji,procent_kompresji



pliki=['clip_1','clip_2','clip_3','clip_4','clip_5']
metody=['444','440','420','422','411','410']

klatka_kluczowa=1
for nazwa in pliki:
    for metoda in metody:
        cap = cv2.VideoCapture(nazwa+'.mp4')
        liczba_klatek=15
        if liczba_klatek<0:
            liczba_klatek=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skompresowane_bez_RLE_Y=[]
        skompresowane_bez_RLE_Cr=[]
        skompresowane_bez_RLE_Cb=[]

        skompresowane_RLE_Y=[]
        skompresowane_RLE_Cr=[]
        skompresowane_RLE_Cb=[]
        for i in range(liczba_klatek):
            if(i%klatka_kluczowa==0):
                ret_klucz, frame_klucz = cap.read()
                Y,Cr,Cb=redukcja_Chromatycznosci(frame_klucz,metoda)
                kom_Y,kom_Cr,kom_Cb=kompresja_bez_RLE(Y,Cr,Cb,frame_klucz)
                

                nowa_macierz_tab_Y,wymiar_Y,stopien_kompresji_Y,procent_kompresji_Y=kompresja_RLE(Y)
                nowa_macierz_tab_Cr,wymiar_Cr,stopien_kompresji_Cr,procent_kompresji_Cr=kompresja_RLE(Cr)
                nowa_macierz_tab_Cb,wymiar_Cb,stopien_kompresji_Cb,procent_kompresji_Cb=kompresja_RLE(Cb)
                skompresowane_bez_RLE_Y.append(kom_Y)
                skompresowane_bez_RLE_Cr.append(kom_Cr)
                skompresowane_bez_RLE_Cb.append(kom_Cb)
                skompresowane_RLE_Y.append(procent_kompresji_Y)
                skompresowane_RLE_Cr.append(procent_kompresji_Cr)
                skompresowane_RLE_Cb.append(procent_kompresji_Cb)
                kw_Y,kw_Cr,kw_Cb=kwantyzacja(Y,Cr,Cb)
                dkw_Y,dkw_Cr,dkw_Cb=dekwantyzacja(kw_Y,kw_Cr,kw_Cb)
                Y2,Cr2,Cb2=redukcja_Chromatycznosci_do_YCrCb(dkw_Y,dkw_Cr,dkw_Cb)
                Y2=np.clip(Y2,0,255)
                Cr2=np.clip(Cr2,0,255)
                Cb2=np.clip(Cb2,0,255)
                nowy_obraz=np.dstack([Y2,Cr2,Cb2]).astype(np.uint8)
            else:
                ret, frame = cap.read()
                diff=frame.astype(float)-frame_klucz.astype(float)

                Y,Cr,Cb=redukcja_Chromatycznosci(diff,metoda)
                kw_Y,kw_Cr,kw_Cb=kwantyzacja(Y, Cr, Cb)
                dkw_Y,dkw_Cr,dkw_Cb=dekwantyzacja(kw_Y,kw_Cr,kw_Cb)
                chroma_Y,chroma_Cr,chroma_Cb=redukcja_Chromatycznosci_do_YCrCb(dkw_Y,dkw_Cr,dkw_Cb)
                chroma_Y=np.clip(chroma_Y,0,255)
                chroma_Cr=np.clip(chroma_Cr,0,255)
                chroma_Cb=np.clip(chroma_Cb,0,255)
                nowy_obraz=np.dstack([chroma_Y,chroma_Cr,chroma_Cb]).astype(np.uint8)
                jpge=YCrCb_do_RGB(nowy_obraz)

                kom_Y,kom_Cr,kom_Cb=kompresja_bez_RLE(Y,Cr,Cb,diff)
                

                nowa_macierz_tab_Y,wymiar_Y,stopien_kompresji_Y,procent_kompresji_Y=kompresja_RLE(jpge[:,:,0])
                nowa_macierz_tab_Cr,wymiar_Cr,stopien_kompresji_Cr,procent_kompresji_Cr=kompresja_RLE(jpge[:,:,1])
                nowa_macierz_tab_Cb,wymiar_Cb,stopien_kompresji_Cb,procent_kompresji_Cb=kompresja_RLE(jpge[:,:,2])
                skompresowane_bez_RLE_Y.append(kom_Y)
                skompresowane_bez_RLE_Cr.append(kom_Cr)
                skompresowane_bez_RLE_Cb.append(kom_Cb)
                skompresowane_RLE_Y.append(procent_kompresji_Y)
                skompresowane_RLE_Cr.append(procent_kompresji_Cr)
                skompresowane_RLE_Cb.append(procent_kompresji_Cb)
                print(nazwa,i)

            if(i==liczba_klatek-1):
                plt.suptitle('Porównanie ostatniej klatki '+nazwa+' '+metoda)
                plt.subplot(1,2,1)
                plt.title('Oryginał')
                plt.imshow(cv2.cvtColor(frame_klucz,cv2.COLOR_BGR2RGB))
                plt.subplot(1,2,2)
                plt.title('Po kompresji JPEG')
                plt.imshow(cv2.cvtColor(nowy_obraz,cv2.COLOR_BGR2RGB))
                plt.savefig(nazwa+'/porownanie_ostatniej_klatki_'+nazwa+'_'+metoda+'.jpg')
                plt.show()

                plt.suptitle('Porównanie ostatniej klatki '+nazwa+' '+metoda+'\nPrzybliżone')
                plt.subplot(1,2,1)
                plt.title('Oryginał')
                plt.imshow(cv2.cvtColor(frame_klucz,cv2.COLOR_BGR2RGB))
                plt.xlim(225,310)
                plt.ylim(200,130)
                plt.subplot(1,2,2)
                plt.title('Po kompresji JPEG')
                plt.imshow(cv2.cvtColor(nowy_obraz,cv2.COLOR_BGR2RGB))
                plt.xlim(225,310)
                plt.ylim(200,130)
                plt.savefig(nazwa+'/porownanie_ostatniej_klatki_'+nazwa+'_'+metoda+'_przyblizenie.jpg')
                plt.show()


        plt.title('Plik: '+nazwa+' subsumpling: '+metoda+' z RLE')
        plt.plot(skompresowane_RLE_Y,c='b')
        plt.plot(skompresowane_RLE_Cr,c='g')
        plt.plot(skompresowane_RLE_Cb,c='orange')
        plt.legend(['Y','Cr','Cb'])
        plt.ylabel('Zysk pamięci [%]')
        plt.xlabel('Numer klatki')
        plt.savefig(nazwa+'/'+nazwa+'subsumpling_'+metoda+'_z_RLE.jpg')
        plt.show()

        plt.title('Plik: '+nazwa+' subsumpling: '+metoda+' bez RLE')
        plt.plot(skompresowane_bez_RLE_Y,c='b')
        plt.plot(skompresowane_bez_RLE_Cr,c='g')
        plt.plot(skompresowane_bez_RLE_Cb,c='orange')
        plt.legend(['Y','Cr','Cb'])
        plt.ylabel('Zysk pamięci [%]')
        plt.xlabel('Numer klatki')
        plt.savefig(nazwa+'/'+nazwa+'_subsumpling_'+metoda+'_bez_RLE.jpg')
        plt.show()
