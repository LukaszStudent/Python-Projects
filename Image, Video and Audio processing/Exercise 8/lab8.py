import cv2
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt


def RGB_do_YCrCb(obraz):
    nowy_obraz=cv2.cvtColor(obraz,cv2.COLOR_RGB2YCrCb).astype(int)
    return nowy_obraz

def YCrCb_do_RGB(obraz):
    nowy_obraz=cv2.cvtColor(obraz.astype(np.uint8),cv2.COLOR_YCrCb2RGB)
    return nowy_obraz


def redukcja_Chromatycznosci(obraz,metoda):
    Y=obraz[:,:,0]
    if(metoda=='4:4:4'):
        Cr=obraz[:,:,1]
        Cb=obraz[:,:,2]
    elif(metoda=='4:4:0'):
        Cr=obraz[0::2,:,1]
        Cb=obraz[0::2,:,2]
    elif(metoda=='4:2:0'):
        Cr=obraz[0::2,0::2,1]
        Cb=obraz[0::2,0::2,2]
    elif(metoda=='4:2:2'):
        Cr=obraz[:,0::2,1]
        Cb=obraz[:,0::2,2]
    elif(metoda=='4:1:1'):
        Cr=obraz[:,0::4,1]
        Cb=obraz[:,0::4,2]
    elif(metoda=='4:1:0'):
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

def zigzag(A):
    template= n= np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    if len(A.shape)==1:
        B=np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                B[r,c]=A[template[r,c]]
    else:
        B=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):
                B[template[r,c]]=A[r,c]
    return B

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


def main(obraz,metoda):
    nowa_macierz_tab,wymiar,stopien_kompresji,procent_kompresji=kompresja_RLE(obraz)
    plt.title('Obraz wejściowy\nstopień kopresji: '+str(stopien_kompresji)+', czyli '+str(procent_kompresji)+'%')
    plt.imshow(cv2.cvtColor(obraz,cv2.COLOR_BGR2RGB))
    plt.savefig('obraz_wejsciowy_rle.jpg')
    plt.show()

    ycrcb=RGB_do_YCrCb(obraz)
    rgb=YCrCb_do_RGB(ycrcb)
    #Plotowanie konwersji z RGB do YCrCB i na odwrot
    plt.suptitle('RGB do YCrCb')
    plt.subplot(1,2,1)
    plt.title('RGB')
    plt.imshow(cv2.cvtColor(obraz,cv2.COLOR_BGR2RGB))
    plt.subplot(1,2,2)
    plt.title('YCrCb')
    plt.imshow(cv2.cvtColor(obraz,cv2.COLOR_BGR2YCrCb))
    plt.savefig('rgb_do_ycrcb.jpg')
    plt.show()


    plt.suptitle('YCrCb do RGB')
    plt.subplot(1,2,1)
    plt.title('YCrCb')
    plt.imshow(cv2.cvtColor(obraz,cv2.COLOR_BGR2YCrCb))
    plt.subplot(1,2,2)
    plt.title('RGB')
    plt.imshow(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
    plt.savefig('ycrcb_do_rgb.jpg')
    plt.show()


    Y, Cr, Cb=redukcja_Chromatycznosci(obraz,metoda)
    #Plotowanie zredukowanych obrazow
    plt.subplot(1,3,1)
    plt.suptitle('Redukcja chromatyczności')
    plt.title('Kanał Y')
    plt.imshow(cv2.cvtColor(Y,cv2.COLOR_BGR2RGB))
    plt.subplot(1,3,2)
    plt.title('Kanał Cr')
    plt.imshow(cv2.cvtColor(Cr,cv2.COLOR_BGR2RGB))
    plt.subplot(1,3,3)
    plt.title('Kanał Cb')
    plt.imshow(cv2.cvtColor(Cb,cv2.COLOR_BGR2RGB))
    plt.savefig('redukcja_chromatycznosci.jpg')
    plt.show()


    stary_Y,stary_Cr,stary_Cb = redukcja_Chromatycznosci_do_YCrCb(Y,Cr,Cb)
    #Plotowanie odwzorowanych YCrCb
    plt.subplot(1,3,1)
    plt.suptitle('Odwzorowane kanały')
    plt.title('Kanał Y')
    plt.imshow(cv2.cvtColor(stary_Y.astype(np.uint8),cv2.COLOR_GRAY2RGB))
    plt.subplot(1,3,2)
    plt.title('Kanał Cr')
    plt.imshow(cv2.cvtColor(stary_Cr.astype(np.uint8),cv2.COLOR_GRAY2RGB))
    plt.subplot(1,3,3)
    plt.title('Kanał Cb')
    plt.imshow(cv2.cvtColor(stary_Cb.astype(np.uint8),cv2.COLOR_GRAY2RGB))
    plt.savefig('redukcja_chromatycznosci_z_powrotem.jpg')
    plt.show()


    #Kwantyzacja
    kw_Y,kw_Cr,kw_Cb=kwantyzacja(Y, Cr, Cb)
    plt.suptitle('Kwantyzacja')
    plt.subplot(1,3,1)
    plt.title('Kanał Y')
    plt.imshow(cv2.cvtColor(kw_Y.astype(np.uint8),cv2.COLOR_GRAY2RGB))
    plt.subplot(1,3,2)
    plt.title('Kanał Cr')
    plt.imshow(cv2.cvtColor(kw_Cr.astype(np.uint8),cv2.COLOR_GRAY2RGB))
    plt.subplot(1,3,3)
    plt.title('Kanał Cb')
    plt.imshow(cv2.cvtColor(kw_Cb.astype(np.uint8),cv2.COLOR_GRAY2RGB))
    plt.savefig('kwantyzacja.jpg')
    plt.show()


    #Dekwantyzacja
    dkw_Y,dkw_Cr,dkw_Cb=dekwantyzacja(kw_Y,kw_Cr,kw_Cb)
    plt.suptitle('Dekwantyzacja')
    plt.subplot(1,3,1)
    plt.title('Kanał Y')
    plt.imshow(dkw_Y,cmap=plt.cm.gray)
    plt.subplot(1,3,2)
    plt.title('Kanał Cr')
    plt.imshow(dkw_Cr,cmap=plt.cm.gray)
    plt.subplot(1,3,3)
    plt.title('Kanał Cb')
    plt.imshow(dkw_Cb,cmap=plt.cm.gray)
    plt.savefig('dekwantyzacja.jpg')
    plt.show()


    chroma_Y,chroma_Cr,chroma_Cb=redukcja_Chromatycznosci_do_YCrCb(dkw_Y,dkw_Cr,dkw_Cb)
    plt.suptitle('Powrót do YCrCb')
    plt.subplot(1,3,1)
    plt.title('Kanał Y')
    plt.imshow(chroma_Y,cmap=plt.cm.gray)
    plt.subplot(1,3,2)
    plt.title('Kanał Cr')
    plt.imshow(chroma_Cr,cmap=plt.cm.gray)
    plt.subplot(1,3,3)
    plt.title('Kanał Cb')
    plt.imshow(chroma_Cb,cmap=plt.cm.gray)
    plt.savefig('powrot_do_warst_ycrcb.jpg')
    plt.show()

    chroma_Y=np.clip(chroma_Y,0,255)
    chroma_Cr=np.clip(chroma_Cr,0,255)
    chroma_Cb=np.clip(chroma_Cb,0,255)

    nowy_obraz=np.dstack([chroma_Y,chroma_Cr,chroma_Cb]).astype(np.uint8)
    plt.title('Połączone warstwy do palety YCrCb')
    plt.imshow(cv2.cvtColor(nowy_obraz,cv2.COLOR_BGR2YCrCb))
    plt.savefig('polaczone_z_powrotem_do_ycrcb.jpg')
    plt.show()

    jpge=YCrCb_do_RGB(nowy_obraz)
    nowa_macierz_tab,wymiar,stopien_kompresji,procent_kompresji=kompresja_RLE(jpge)
    plt.title('Obraz wynikowy\nstopień kopresji: '+str(stopien_kompresji)+', czyli '+str(procent_kompresji)+'%')
    plt.imshow(cv2.cvtColor(nowy_obraz,cv2.COLOR_BGR2RGB))
    plt.savefig('obraz_wynikowy.jpg')
    plt.show()

    plt.subplots_adjust(wspace=0.8,hspace=0.8)
    plt.subplot(4,2,1)
    plt.imshow(cv2.cvtColor(obraz,cv2.COLOR_BGR2RGB))
    plt.xlim(128,256)
    plt.ylim(128,256)
    plt.subplot(4,2,2)
    plt.imshow(cv2.cvtColor(nowy_obraz,cv2.COLOR_BGR2RGB))
    plt.xlim(128,256)
    plt.ylim(128,256)
    plt.subplot(4,2,3)
    plt.imshow(Y,cmap=plt.cm.gray)
    plt.xlim(128,256)
    plt.ylim(128,256)
    plt.subplot(4,2,4)
    plt.imshow(chroma_Y,cmap=plt.cm.gray)
    plt.xlim(128,256)
    plt.ylim(128,256)
    plt.subplot(4,2,5)
    plt.imshow(Cr,cmap=plt.cm.gray)
    plt.xlim(128,256)
    plt.ylim(128,256)
    plt.subplot(4,2,6)
    plt.imshow(Cb,cmap=plt.cm.gray)
    plt.xlim(128,256)
    plt.ylim(128,256)
    plt.subplot(4,2,7)
    plt.imshow(chroma_Cr,cmap=plt.cm.gray)
    plt.xlim(128,256)
    plt.ylim(128,256)
    plt.subplot(4,2,8)
    plt.imshow(chroma_Cb,cmap=plt.cm.gray)
    plt.xlim(128,256)
    plt.ylim(128,256)
    plt.savefig('porownanie_wycinkow.jpg')
    plt.show()


if __name__ == "__main__":
    obraz=cv2.imread('0013.jpg')
    main(obraz,'4:2:0')
