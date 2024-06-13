import numpy as np
import matplotlib.pyplot as plt
import cv2

auto=plt.imread('auto.jpg')
obrazek=plt.imread('lena.jpg')
twarz=plt.imread('0009.png')
slonko=plt.imread('slonko.jpg')
twarz=cv2.cvtColor(twarz,cv2.COLOR_BGR2GRAY)
print(auto.shape)


def colorFit(wartosc_koloru,paleta_kolorow):
    najblizsza_wartosc=np.abs(paleta_kolorow-wartosc_koloru)
    if(len(paleta_kolorow.shape)>2):
        najblizsza_wartosc=np.abs(paleta_kolorow-wartosc_koloru).reshape(-1,3)
        paleta_kolorow_2=paleta_kolorow.reshape(-1,3)
        return paleta_kolorow_2[np.argmin(np.linalg.norm(najblizsza_wartosc,axis=1))]
    else:
        paleta_kolorow=paleta_kolorow.reshape(-1)
    return paleta_kolorow[np.argmin(np.linalg.norm(najblizsza_wartosc,axis=1))]



#DITHERING
#ZADANIE 2
def dithering_losowy(obraz):
    nowy_obraz=np.zeros((obraz.shape),dtype=int)
    for i in range(obraz.shape[0]):
        for j in range(obraz.shape[1]):
            if(obraz[i][j]>np.random.rand()):
                nowy_obraz[i][j]=1

    return nowy_obraz



def dithering_zorganizowany(obraz):
    M2=[[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]]
    M2=np.array(M2).reshape((4,4))
    obraz=(obraz/obraz.max())*15
    nowa_obraz=np.ones((obraz.shape))
    if(len(obraz.shape)>2):
        for i in range(obraz.shape[0]):
            for j in range(obraz.shape[1]):
                for k in range(obraz.shape[2]):
                    if(obraz[i][j][k]>M2[i%M2.shape[0]][j%M2.shape[0]]):
                        pass
                    else:
                        nowa_obraz[i][j][k]=0
    else:
        for i in range(obraz.shape[0]):
            for j in range(obraz.shape[1]):
                if(obraz[i][j]>M2[i%M2.shape[0]][j%M2.shape[0]]):
                    #nowa_obraz[i][j]=M2[i%M2.shape[0]][j%M2.shape[0]]
                    pass
                else:
                    nowa_obraz[i][j]=0

    return nowa_obraz


def dithering_Floyd_Stainberg(obraz,nowa_paleta):
    kopia=obraz.copy()
    if(len(obraz.shape)>2):
        nowa_obraz=np.ones((obraz.shape),dtype=int)
        for i in range(obraz.shape[0]-1):
            for j in range(obraz.shape[1]-1):
                oldpixel = kopia[i][j].copy()
                newpixel = colorFit(oldpixel,nowa_paleta)
                kopia[i][j] = newpixel
                quant_error = oldpixel - newpixel
                nowa_obraz[i][j] = ((kopia[i+1][j]+quant_error*7/16)+(kopia[i-1][j+1]+quant_error*3/16)+(kopia[i][j+1]+quant_error*5/16)+(kopia[i+1][j+1]+quant_error*1/16))/4
    else:
        nowa_obraz=np.ones((obraz.shape),dtype=int)
        for i in range(obraz.shape[0]-1):
            for j in range(obraz.shape[1]-1):
                oldpixel = kopia[i][j]
                newpixel = colorFit(oldpixel,nowa_paleta)
                kopia[i][j] = newpixel
                quant_error = oldpixel - newpixel
                nowa_obraz[i][j] = (kopia[i+1][j]+quant_error*7/16)+(kopia[i-1][j+1]+quant_error*3/16)+(kopia[i][j+1]+quant_error*5/16)+(kopia[i+1][j+1]+quant_error*1/16)
                # nowa_obraz[i + 1][j    ] = kopia[i + 1][j    ] + quant_error * (7 / 16)
                # nowa_obraz[i - 1][j + 1] = kopia[i - 1][j + 1] + quant_error * (3 / 16)
                # nowa_obraz[i    ][j + 1] = kopia[i    ][j + 1] + quant_error * (5 / 16)
                # nowa_obraz[i + 1][j + 1] = kopia[i + 1][j + 1] + quant_error * (1 / 16)
    return nowa_obraz


def kwantyzacja(obraz,liczba_bitow):
    zakresy=np.linspace(0,255,2**liczba_bitow,dtype=int)
    czarny=np.linspace(0,1,2**liczba_bitow)
    if(len(obraz.shape)>2):
        nowy_obraz=np.zeros((obraz.shape),dtype=int)
        for i in range(obraz.shape[0]):
            for j in range(obraz.shape[1]):
                for k in range(obraz.shape[2]):
                    najblizsza_wartosc_idx=(np.abs(zakresy-obraz[i][j][k])).argmin()
                    nowy_obraz[i][j][k]=zakresy[najblizsza_wartosc_idx]
        
    else:
        nowy_obraz=np.zeros((obraz.shape),dtype=float)
        for i in range(obraz.shape[0]):
            for j in range(obraz.shape[1]):
                najblizsza_wartosc_idx=(np.abs(czarny-obraz[i][j])).argmin()
                nowy_obraz[i][j]=czarny[najblizsza_wartosc_idx]

    return nowy_obraz

def obraz_dopaswoany(obraz,paleta):
    if(len(obraz.shape)>2):
        nowy_obraz=np.ones((obraz.shape),dtype=np.uint8)
        for i in range(obraz.shape[0]):
            for j in range(obraz.shape[1]):
                #for k in range(obraz.shape[2]):
                nowy_obraz[i][j]=colorFit(obraz[i][j],paleta)
    else:
        nowy_obraz=np.zeros((obraz.shape))
        for i in range(obraz.shape[0]):
            for j in range(obraz.shape[1]):
                nowy_obraz[i][j]=colorFit(obraz[i][j],paleta)
    return nowy_obraz
    
def main():
    obraz_przetwarzany=auto.copy()
    kw8=kwantyzacja(obraz_przetwarzany,8)
    kw7=kwantyzacja(obraz_przetwarzany,7)
    kw6=kwantyzacja(obraz_przetwarzany,6)
    kw5=kwantyzacja(obraz_przetwarzany,5)
    kw4=kwantyzacja(obraz_przetwarzany,4)
    kw3=kwantyzacja(obraz_przetwarzany,3)
    kw2=kwantyzacja(obraz_przetwarzany,2)
    kw1=kwantyzacja(obraz_przetwarzany,1)

    plt.suptitle('Kwantyzacja obrazu do:')
    plt.subplot(2,4,1)
    plt.title('8 bitów')
    plt.imshow(kw8,cmap=plt.cm.gray)
    plt.subplot(2,4,2)
    plt.title('7 bitów')
    plt.imshow(kw7,cmap=plt.cm.gray)
    plt.subplot(2,4,3)
    plt.title('6 bitów')
    plt.imshow(kw6,cmap=plt.cm.gray)
    plt.subplot(2,4,4)
    plt.title('5 bitów')
    plt.imshow(kw5,cmap=plt.cm.gray)
    plt.subplot(2,4,5)
    plt.title('4 bitów')
    plt.imshow(kw4,cmap=plt.cm.gray)
    plt.subplot(2,4,6)
    plt.title('3 bitów')
    plt.imshow(kw3,cmap=plt.cm.gray)
    plt.subplot(2,4,7)
    plt.title('2 bitów')
    plt.imshow(kw2,cmap=plt.cm.gray)
    plt.subplot(2,4,8)
    plt.title('1 bitu')
    plt.imshow(kw1,cmap=plt.cm.gray)
    plt.savefig('kwantyzacja.jpg')
    plt.show()


    poziom_kwantyzacji=kw4.copy()
    paleta_8bit=np.array([[0, 0, 0,],[0, 0, 1,],[0, 1, 0,],
                            [0, 1, 1,],[1, 0, 0,],[1, 0, 1,],
                            [1, 1, 0,],[1, 1, 1,]])
    paleta_16bit=np.array([[0,0,0],[0,1,1],[0,0,1],
                            [1,0,1],[0,0.5,0],[0.5,0.5,0.5],
                            [0,1,0],[0.5,0,0],[0,0,0.5],
                            [0.5,0.5,0],[0.5,0,0.5],[1,0,0]
                            ,[0.75,0.75,0.75],[0,0.5,0.5],
                            [1,1,1],[1,1,0]])
    plt.suptitle('Porównanie metod')
    plt.subplot(1,4,1)
    plt.title('Oryginał')
    plt.imshow(obraz_przetwarzany,cmap=plt.cm.gray)
    plt.subplot(1,4,2)
    if(len(obraz_przetwarzany.shape)>2):
        plt.title('Dopasowany\ndo palety')
        paleta_auto=np.array(([28,25,19],[140,0,11],[169,149,37],[0,0,0],[144,135,118],[255,255,255]))
        paleta_auto=paleta_auto.reshape((1,6,3))
        dopasowany=obraz_dopaswoany(obraz_przetwarzany,paleta_auto)
        plt.imshow(dopasowany)
    else:
        plt.title('Dithering\nlosowy')
        losowy=dithering_losowy(poziom_kwantyzacji)
        plt.imshow(losowy,cmap=plt.cm.gray)
    plt.subplot(1,4,3)
    plt.title('Dithering\nzorganizowany')
    zorganizowany=dithering_zorganizowany(poziom_kwantyzacji)
    plt.imshow(zorganizowany,cmap=plt.cm.gray)
    plt.subplot(1,4,4)
    plt.title('Dithering\nFloyd-Steinberg')
    floyd=dithering_Floyd_Stainberg(poziom_kwantyzacji,obraz_przetwarzany.copy())
    plt.imshow(floyd,cmap=plt.cm.gray)
    plt.savefig('dithering.jpg')
    plt.show()


if __name__ ==main():
    main()