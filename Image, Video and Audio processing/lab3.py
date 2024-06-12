import numpy as np 
import matplotlib.pyplot as plt 
import cv2

obrazek=plt.imread('obrazek.jpg')
#ZADANIE 1 - IMPLEMENTACJA ALGORYTMOW POWIEKSZANIA I POMNIEJSZANIA OBRAZU
#POWIEKSZANIE OBRAZU
#Metoda najblizszych sasiadow
def najblizszych_sasiadow(obraz,ratio):
    powiekszenie=(ratio/100)
    nowy_rozmiar_y=int((obraz.shape[0]*(ratio))/100)
    nowy_rozmiar_x=int((obraz.shape[1]*(ratio))/100)
    nowy_wymiar=int(obraz.shape[2])
    nowy_obraz=np.zeros((nowy_rozmiar_y,nowy_rozmiar_x,nowy_wymiar),dtype=np.uint8)

    for i in range(nowy_rozmiar_y):
        for j in range(nowy_rozmiar_x):
            for k in range(nowy_wymiar):
                wiersz=int(np.floor(i/powiekszenie))
                kolumna=int(np.floor(j/powiekszenie))
                nowy_obraz[i][j][k]=obraz[wiersz][kolumna][k]

    return nowy_obraz

#Metoda Interpolacji Dwuliniowej
def interpolacja_dwuliniowa(obraz,ratio):
    powiekszenie=ratio/100
    nowy_rozmiar_y=int((obraz.shape[0]*(ratio))/100)
    nowy_rozmiar_x=int((obraz.shape[1]*(ratio))/100)
    nowy_wymiar=int(obraz.shape[2])
    nowy_obraz=np.zeros((nowy_rozmiar_y,nowy_rozmiar_x,nowy_wymiar),dtype=np.uint8)

    for i in range(0,int(nowy_rozmiar_y-powiekszenie),1):
        for j in range(0,int(nowy_rozmiar_x-powiekszenie),1):
            for k in range(nowy_wymiar):
                if(j%powiekszenie!=0 and i%powiekszenie==0):
                    nowy_obraz[i][j][k]=((j%powiekszenie)/powiekszenie)*obraz[int(i/powiekszenie)+1][int(j/powiekszenie)][k]+((powiekszenie-(j%powiekszenie))/powiekszenie)*obraz[int(i/powiekszenie)][int(j/powiekszenie)][k]
                elif(i%powiekszenie!=0):
                    nowy_obraz[i][j][k]=((i%powiekszenie)/powiekszenie)*obraz[int(i/powiekszenie)][int(j/powiekszenie)+1][k]+((powiekszenie-(i%powiekszenie))/powiekszenie)*obraz[int(i/powiekszenie)][int(j/powiekszenie)][k]
                else:
                    nowy_obraz[i][j][k]=obraz[int(i/powiekszenie)][int(j/powiekszenie)][k]

    return nowy_obraz

#ZMNIEJSZANIE OBRAZU
#Srednia
def pomniejszanie_obrazu_srednia(obraz,ratio):
    pomniejszenie=100/ratio
    nowy_rozmiar_y=int((obraz.shape[0]*(ratio))/100)
    nowy_rozmiar_x=int((obraz.shape[1]*(ratio))/100)
    nowy_wymiar=int(obraz.shape[2])
    nowy_obraz=np.zeros((nowy_rozmiar_y,nowy_rozmiar_x,nowy_wymiar),dtype=np.uint8)

    for i in range(0,nowy_rozmiar_y-1,1):
        for j in range(0,nowy_rozmiar_x-1,1):
            for k in range(nowy_wymiar):
                nowy_obraz[i][j][k]=np.mean([obraz[int(i*pomniejszenie-1)][int(j*pomniejszenie-1)][k],
                                        obraz[int(i*pomniejszenie)][int(j*pomniejszenie-1)][k],
                                        obraz[int(i*pomniejszenie+1)][int(j*pomniejszenie-1)][k],
                                        obraz[int(i*pomniejszenie-1)][int(j*pomniejszenie)][k],
                                        obraz[int(i*pomniejszenie)][int(j*pomniejszenie)][k],
                                        obraz[int(i*pomniejszenie+1)][int(j*pomniejszenie)][k],
                                        obraz[int(i*pomniejszenie-1)][int(j*pomniejszenie+1)][k],
                                        obraz[int(i*pomniejszenie)][int(j*pomniejszenie+1)][k],
                                        obraz[int(i*pomniejszenie+1)][int(j*pomniejszenie+1)][k]])

    return nowy_obraz

#Mediana
def pomniejszanie_obrazu_mediana(obraz,ratio):
    pomniejszenie=100/ratio
    nowy_rozmiar_y=int((obraz.shape[0]*(ratio))/100)
    nowy_rozmiar_x=int((obraz.shape[1]*(ratio))/100)
    nowy_wymiar=int(obraz.shape[2])
    nowy_obraz=np.zeros((nowy_rozmiar_y,nowy_rozmiar_x,nowy_wymiar),dtype=np.uint8)

    for i in range(0,nowy_rozmiar_y-1,1):
        for j in range(0,nowy_rozmiar_x-1,1):
            for k in range(nowy_wymiar):
                nowy_obraz[i][j][k]=np.median([obraz[int(i*pomniejszenie-1)][int(j*pomniejszenie-1)][k],
                                        obraz[int(i*pomniejszenie)][int(j*pomniejszenie-1)][k],
                                        obraz[int(i*pomniejszenie+1)][int(j*pomniejszenie-1)][k],
                                        obraz[int(i*pomniejszenie-1)][int(j*pomniejszenie)][k],
                                        obraz[int(i*pomniejszenie)][int(j*pomniejszenie)][k],
                                        obraz[int(i*pomniejszenie+1)][int(j*pomniejszenie)][k],
                                        obraz[int(i*pomniejszenie-1)][int(j*pomniejszenie+1)][k],
                                        obraz[int(i*pomniejszenie)][int(j*pomniejszenie+1)][k],
                                        obraz[int(i*pomniejszenie+1)][int(j*pomniejszenie+1)][k]])

    return nowy_obraz

#Srednia wazona
def pomniejszanie_obrazu_srednia_wazona(obraz,ratio):
    pomniejszenie=100/ratio
    nowy_rozmiar_y=int((obraz.shape[0]*(ratio))/100)
    nowy_rozmiar_x=int((obraz.shape[1]*(ratio))/100)
    nowy_wymiar=int(obraz.shape[2])
    nowy_obraz=np.zeros((nowy_rozmiar_y,nowy_rozmiar_x,nowy_wymiar),dtype=np.uint8)

    for i in range(0,nowy_rozmiar_y-1,1):
        for j in range(0,nowy_rozmiar_x-1,1):
            for k in range(nowy_wymiar):
                nowy_obraz[i][j][k]=np.average([obraz[int(i*pomniejszenie-1)][int(j*pomniejszenie-1)][k],
                                        obraz[int(i*pomniejszenie)][int(j*pomniejszenie-1)][k],
                                        obraz[int(i*pomniejszenie+1)][int(j*pomniejszenie-1)][k],
                                        obraz[int(i*pomniejszenie-1)][int(j*pomniejszenie)][k],
                                        obraz[int(i*pomniejszenie)][int(j*pomniejszenie)][k],
                                        obraz[int(i*pomniejszenie+1)][int(j*pomniejszenie)][k],
                                        obraz[int(i*pomniejszenie-1)][int(j*pomniejszenie+1)][k],
                                        obraz[int(i*pomniejszenie)][int(j*pomniejszenie+1)][k],
                                        obraz[int(i*pomniejszenie+1)][int(j*pomniejszenie+1)][k]],weights=np.random.rand(9))

    return nowy_obraz


def main():
    #ZADANIE 2 - TESTOWANIE
    #Powiekszanie obrazu
    powiekszenie=150
    knn=najblizszych_sasiadow(obrazek,powiekszenie)
    bilin=interpolacja_dwuliniowa(obrazek,powiekszenie)
    plt.subplot(2,3,1)
    plt.title('Oryginał')
    plt.imshow(obrazek)
    plt.subplot(2,3,2)
    plt.title('Najbliższych\nsąsiadów')
    plt.imshow(knn)
    plt.subplot(2,3,3)
    plt.title('Interpolacja\ndwuliniowa')
    plt.imshow(bilin)

    plt.subplot(2,3,4)
    plt.xlim(0,15)
    plt.ylim(0,15)
    plt.imshow(obrazek)
    plt.subplot(2,3,5)
    plt.xlim(0,15*powiekszenie/100)
    plt.ylim(0,15*powiekszenie/100)
    plt.imshow(knn)
    plt.subplot(2,3,6)
    plt.xlim(0,15*powiekszenie/100)
    plt.ylim(0,15*powiekszenie/100)
    plt.imshow(bilin)
    plt.savefig('porownanie_powiekszen.jpg')
    plt.show()

    #Pomniejszanie obrazu
    pomniejszenie=50
    przyblizenie=10
    srednia=pomniejszanie_obrazu_srednia(obrazek,pomniejszenie)
    mediana=pomniejszanie_obrazu_mediana(obrazek,pomniejszenie)
    srednia_wazona=pomniejszanie_obrazu_srednia_wazona(obrazek,pomniejszenie)

    plt.subplot(2,4,1)
    plt.title('Oryginał')
    plt.imshow(obrazek)
    plt.subplot(2,4,2)
    plt.title('Średnia')
    plt.imshow(srednia)
    plt.subplot(2,4,3)
    plt.title('Mediana')
    plt.imshow(mediana)
    plt.subplot(2,4,4)
    plt.title('Średnia ważona')
    plt.imshow(srednia_wazona)

    plt.subplot(2,4,5)
    plt.xlim(0,przyblizenie)
    plt.ylim(0,przyblizenie)
    plt.imshow(obrazek)
    plt.subplot(2,4,6)
    plt.xlim(0,przyblizenie/(100/pomniejszenie))
    plt.ylim(0,przyblizenie/(100/pomniejszenie))
    plt.imshow(srednia)
    plt.subplot(2,4,7)
    plt.xlim(0,przyblizenie/(100/pomniejszenie))
    plt.ylim(0,przyblizenie/(100/pomniejszenie))
    plt.imshow(mediana)
    plt.subplot(2,4,8)
    plt.xlim(0,przyblizenie/(100/pomniejszenie))
    plt.ylim(0,przyblizenie/(100/pomniejszenie))
    plt.imshow(srednia_wazona)
    plt.savefig('porownanie_pomniejszen.jpg')
    plt.show()


    #Kontury powiekszonych obrazow
    plt.subplot(2,3,1)
    plt.title('Oryginał')
    plt.imshow(obrazek)
    plt.subplot(2,3,2)
    plt.title('Najbliższych\nsąsiadów')
    plt.imshow(knn)
    plt.subplot(2,3,3)
    plt.title('Interpolacja\ndwuliniowa')
    plt.imshow(bilin)

    plt.subplot(2,3,4)
    edges_oryginalny=cv2.Canny(obrazek,obrazek.shape[0],obrazek.shape[1])
    plt.imshow(edges_oryginalny)
    plt.subplot(2,3,5)
    edges_knn=cv2.Canny(knn,knn.shape[0],knn.shape[1])
    plt.imshow(edges_knn)
    plt.subplot(2,3,6)
    edges_bilin=cv2.Canny(bilin,bilin.shape[0],bilin.shape[1])
    plt.imshow(edges_bilin)
    plt.savefig('wykrywanie_krawedzi_powiekszenie.jpg')
    plt.show()

    #Kontury pomniejszonych obrazow
    plt.subplot(2,4,1)
    plt.title('Oryginał')
    plt.imshow(obrazek)
    plt.subplot(2,4,2)
    plt.title('Średnia')
    plt.imshow(srednia)
    plt.subplot(2,4,3)
    plt.title('Mediana')
    plt.imshow(mediana)
    plt.subplot(2,4,4)
    plt.title('Średnia ważona')
    plt.imshow(srednia_wazona)

    plt.subplot(2,4,5)
    plt.imshow(edges_oryginalny)
    plt.subplot(2,4,6)
    edges_srednia=cv2.Canny(srednia,srednia.shape[0],srednia.shape[1])
    plt.imshow(edges_srednia)
    plt.subplot(2,4,7)
    edges_mediana=cv2.Canny(mediana,mediana.shape[0],mediana.shape[1])
    plt.imshow(edges_mediana)
    plt.subplot(2,4,8)
    edges_srednia_wazona=cv2.Canny(srednia_wazona,srednia_wazona.shape[0],srednia_wazona.shape[1])
    plt.imshow(edges_srednia_wazona)
    plt.savefig('wykrywanie_krawedzi_pomniejszenie.jpg')
    plt.show()


if __name__ ==main():
    main()