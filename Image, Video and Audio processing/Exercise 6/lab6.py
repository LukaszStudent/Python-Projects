import numpy as np
import cv2
import matplotlib.pyplot as plt


skan=plt.imread('rysunek.jpg')
skan=cv2.cvtColor(skan,cv2.COLOR_RGB2GRAY)
techniczny=plt.imread('RysunekTechniczny.jpg')
obrazek=plt.imread('Obrazek.jpg')

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
    return (nowa_macierz_tab,macierz.shape,stopien_kompresji,procent_kompresji)


def dekompresja_RLE(zkoompresowana_wiadomosc):
    wiadomosc=zkoompresowana_wiadomosc[0]
    oryginalna_wiadomosc_tab=[]
    for i in range(0,len(wiadomosc),2):
        for j in range(wiadomosc[i]):
            oryginalna_wiadomosc_tab.append(wiadomosc[i+1])

    return oryginalna_wiadomosc_tab


def kompresja_ByteRun(macierz):
    nowa_macierz=macierz.reshape(-1)
    licznik=1
    licznik_roznych=1
    nowa_macierz_tab=[]

    for i in range(len(nowa_macierz)):        
        if(i<len(nowa_macierz)-1 and nowa_macierz[i]==nowa_macierz[i+1]):
            licznik+=1
            if(licznik_roznych!=1):
                licznik_roznych=licznik_roznych-2
                nowa_macierz_tab.append([(licznik_roznych-1),nowa_macierz[i-licznik_roznych:i].tolist()])
                licznik_roznych=1
        
        if(i<len(nowa_macierz)-1 and nowa_macierz[i]!=nowa_macierz[i+1]):
            if(licznik!=1):
                nowa_macierz_tab.append([(licznik-1)*(-1),nowa_macierz[i]])
                licznik=1
            licznik_roznych+=1

        if(i==len(nowa_macierz)-1 and licznik_roznych!=1):
            licznik_roznych=licznik_roznych-2
            nowa_macierz_tab.append([len(nowa_macierz[i-licznik_roznych:])-1,nowa_macierz[i-licznik_roznych:].tolist()])

        if(i==len(nowa_macierz)-1 and licznik!=1):
            licznik_roznych=licznik_roznych-2
            nowa_macierz_tab.append([(licznik-1)*(-1),nowa_macierz[i]])


    nowa_macierz_tab=list(np.concatenate(nowa_macierz_tab).flat)
    poprawna_macierz=[]
    for i in range(len(nowa_macierz_tab)):
        if(isinstance(nowa_macierz_tab[i],int)):
            poprawna_macierz.append(nowa_macierz_tab[i])
        else:
            for j in range(len(nowa_macierz_tab[i])):
                poprawna_macierz.append(nowa_macierz_tab[i][j])

    nowa_macierz_tab=np.array(poprawna_macierz).reshape(-1)


    stopien_kompresji=len(nowa_macierz)/len(nowa_macierz_tab)
    procent_kompresji=100*1/stopien_kompresji
    return (nowa_macierz_tab,macierz.shape,stopien_kompresji,procent_kompresji)


def dekompresja_ByteRun(zkoompresowana_wiadomosc):
    wiadomosc=zkoompresowana_wiadomosc
    oryginalna_wiadomosc_tab=[]
    for i in range(len(wiadomosc)-1):
        if(wiadomosc[i]<0):
            liczba_tych_samych=wiadomosc[i]*(-1)+1
            for j in range(liczba_tych_samych):
                oryginalna_wiadomosc_tab.append(wiadomosc[i+1])  
        if(i==len(wiadomosc)-2):
            break    
        i+=2
        if(wiadomosc[(i-2)]<=0):
            liczba_roznych=wiadomosc[i]+1
            for j in range(liczba_roznych):
                if(i+j+1<len(wiadomosc)-1):
                    oryginalna_wiadomosc_tab.append(wiadomosc[i+j+1])
                else:
                    break
            
    return oryginalna_wiadomosc_tab


def QuadTree(macierz):
    global unikalne_macierze,rozmiar_macierzy,wymiar_macierzy
    # macierz=np.uint8(macierz)
    # if(len(wymiar_macierzy)>2):
    #     macierz=cv2.cvtColor(macierz,cv2.COLOR_BGR2RGB)
    #     macierz=cv2.cvtColor(macierz,cv2.COLOR_RGB2GRAY)

    macierz_00=macierz[0:int(np.floor(macierz.shape[0]/2)),0:int(np.ceil(macierz.shape[1]/2))]
    macierz_01=macierz[0:int(np.floor(macierz.shape[0]/2)):,int(np.ceil(macierz.shape[1]/2)):]
    macierz_10=macierz[int(np.floor(macierz.shape[0]/2)):,0:int(np.ceil(macierz.shape[1]/2))]
    macierz_11=macierz[int(np.floor(macierz.shape[0]/2)):,int(np.ceil(macierz.shape[1]/2)):]

    cztery_macierze=[macierz_00,macierz_01,macierz_10,macierz_11]
    rozne_macierze=[]
    for macierz in cztery_macierze:
        if(len(np.unique(macierz))>1):
            rozne_macierze.append(macierz)
        else:
            unikalne_macierze.append(macierz.tolist())

    for i in rozne_macierze:
        QuadTree(i)

    stopien_kompresji=rozmiar_macierzy/len(list(filter(None, unikalne_macierze)))
    procent_kompresji=100*1/stopien_kompresji
    return stopien_kompresji,procent_kompresji
    


x=np.array([1,1,1,1,2,1,1,1,1,2,1,1]).reshape((3,4))
#print('Oryginalny wektor: ',x)
badany_obraz=skan
unikalne_macierze=[]
wymiar_macierzy=badany_obraz.shape
rozmiar_macierzy=len(badany_obraz.reshape(-1))


koder_RLE=kompresja_RLE(badany_obraz)
#print('Zakodowana wiadomosc RLE: ',koder_RLE[0])
#dekoder_RLE=dekompresja_RLE(koder_RLE)
#print('Odkodowana wiadomość RLE: ',dekoder_RLE)

#koder_ByteRun=kompresja_ByteRun(badany_obraz)
#print('Zakodowana wiadomość ByteRun: ',koder_ByteRun[0])
#dekoder_ByteRun=dekompresja_ByteRun(koder_ByteRun[0])
#print('Odkodowana wiadomość ByteRun: ',dekoder_ByteRun)

#stopien_quad,procent_quad=QuadTree(badany_obraz)


plt.imshow(cv2.cvtColor(badany_obraz, cv2.COLOR_BGR2RGB))
plt.suptitle('Skan dokumnetu\n')
plt.title('RLE stopień kompresji '+str(round(koder_RLE[2],4))+' ,czyli '+str(round(koder_RLE[3],4))+'%\n')
          #'ByteRun stopień kompresji '+str(round(koder_ByteRun[2],4))+' ,czyli '+str(round(koder_ByteRun[3],4))+'%\n')
          #'QuadTree stopień kompresji '+str(round(stopien_quad,4))+' ,czyli '+str(round(procent_quad,4))+'%')
plt.savefig('kompresja.jpg')
plt.show()
