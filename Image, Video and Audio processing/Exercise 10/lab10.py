import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression

obrazy=['malpa.jpg','muka.jpg','smile.jpg']

# for obraz in obrazy:
#     for i in range(0,55,5):
#         obrazek=plt.imread(obraz)
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), i]
#         result, encimg = cv2.imencode('.jpg', obrazek, encode_param)
#         decimg = cv2.imdecode(encimg, 1)
#         cv2.imwrite(f'{obraz}_{i}.png',cv2.cvtColor(decimg,cv2.COLOR_BGR2RGB))

class Miary():
    def __init__(self,obraz,obraz_mody):
        self.obraz=obraz
        self.obraz_mody=obraz_mody
        self.x=obraz.shape[0]
        self.y=obraz.shape[1]
        self.x_mody=obraz_mody.shape[0]
        self.y_mody=obraz_mody.shape[1]

    def MSE(self):
        blad_MSE=0
        for i in range(self.x-1):
            for j in range(self.y-1):
                blad_MSE+=np.abs(self.obraz[i,j,:]-self.obraz_mody[i,j,:])**2
        blad_MSE=blad_MSE*(1/self.x*self.y)
        return blad_MSE

    def NMSE(self):
        licznik=0
        mianownik=0
        blad_NMSE=0
        for i in range(self.x-1):
            for j in range(self.y-1):
                licznik+=np.abs(self.obraz[i,j,:]-self.obraz_mody[i,j,:])**2
                mianownik+=np.abs(self.obraz_mody[i,j,:])**2
        licznik=licznik*(1/(self.x*self.y))
        mianownik=mianownik*(1/(self.x*self.y))
        blad_NMSE=licznik/mianownik
        return blad_NMSE

    def PSNR(self):
        mse=self.MSE()
        blad_PSNR=10*np.log10((255**2)/mse)
        return blad_PSNR

    def IF(self):
        licznik=0
        mianownik=0
        blad_IF=0
        for i in range(self.x-1):
            for j in range(self.y-1):
                licznik+=(self.obraz[i,j,:]-self.obraz_mody[i,j,:])**2
                mianownik+=self.obraz[i,j,:]*self.obraz_mody[i,j,:]
        blad_IF=1-(licznik/mianownik)
        return blad_IF

    def SSIM(self):
        blad_SSIM=skimage.metrics.structural_similarity(self.obraz,self.obraz_mody,multichannel=True)
        return blad_SSIM
        


df = pd.read_csv('badania.csv')
liczba_badanych=len(df.index)

# #kolumny 1:11 mapla; 12:22 muka; 23:33 smile
x=np.array(['Obraz 1','Obraz 2','Obraz 3','Obraz 4','Obraz 5','sObraz 6','Obraz 7','Obraz 8','Obraz 9','Obraz 10'])
icon=['*','^','v','+','.','s','d','H']
colors=['b','g','r','c','m','y','k','purple']
for i in range(liczba_badanych):
    plt.scatter(x,df.iloc[i, 1:11],label=f'Badany {i+1}',marker=icon[i],c=colors[i])
    plt.scatter(x,df.iloc[i, 11:21],marker=icon[i],c=colors[i])
    plt.scatter(x,df.iloc[i, 21:31],marker=icon[i],c=colors[i])
plt.xticks(rotation=45)
plt.ylabel('Oceny MOS')
plt.legend()
plt.title('Mos dla obrazów - wszystkie oceny')
plt.show()

for i in range(liczba_badanych):
    srednia=[]
    for j in range(1,11):
        wynik=(df.iloc[i,j]+df.iloc[i,j+10]+df.iloc[i,j+20])/3
        srednia.append(wynik)
    plt.scatter(x,srednia,label=f'Badany {i+1}',marker=icon[i],c=colors[i])
plt.xticks(rotation=45)
plt.ylabel('Oceny MOS')
plt.legend()
plt.title('Mos dla obrazów - zagregowanego dla badanego')
plt.show()

srednia_obrazow=[]
for i in range(1,11):
    wynik=0
    for j in range(liczba_badanych):
        wynik+=df.iloc[j,i]
    srednia_obrazow.append(wynik/liczba_badanych)
plt.scatter(x,srednia_obrazow,marker=icon[0],c=colors[0])
plt.xticks(rotation=45)
plt.ylabel('Oceny MOS')
plt.title('Mos dla obrazów - zagregowane')
plt.show()

lista=[]
slope_sum=0
intercept_sum=0
x=np.arange(0,10,1)
for i in range(liczba_badanych):
    y=df.iloc[i, 1:11].to_list()
    plt.plot(x, y, linewidth=0, label=f'Badany {i+1}',marker=icon[i],c=colors[i])
    slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
    slope_sum+=slope
    intercept_sum+=intercept
slope_sum/=liczba_badanych
intercept_sum/=liczba_badanych

plt.plot(x, intercept_sum + slope_sum * x,label='Regresja liniowa')
plt.title('Mos dla obrazów - wszystkie oceny')
plt.ylabel('Oceny MOS')
plt.legend()
plt.show()


for oryginalne in obrazy:
    oryginal=plt.imread(oryginalne)
    for i in range(0,55,5):
        obrazek_mody=plt.imread(f'zmodyfikowane_obrazy/{oryginalne}_{i}.png')
        ob1=Miary(oryginal,obrazek_mody)
        mse=ob1.MSE()
        nmse=ob1.NMSE()
        psnr=ob1.PSNR()
        IF=ob1.IF()
        ssim=ob1.SSIM()

        xyz=np.vstack([mse,nmse,psnr,IF])
        corr_matrix = np.corrcoef(xyz).round(decimals=2)
        fig, ax = plt.subplots()
        im = ax.imshow(corr_matrix)
        im.set_clim(-1, 1)
        ax.grid(False)
        ax.xaxis.set(ticks=(0, 1, 2, 3), ticklabels=('mse','nmse','psnr','IFf'))
        ax.yaxis.set(ticks=(0, 1, 2, 3), ticklabels=('mse','nmse','psnr','IF'))
        ax.set_ylim(3.5, -0.5)
        ax.set_title(f'{oryginalne}_{i}.png')
        for x in range(4):
            for j in range(4):
                ax.text(j, x, corr_matrix[x, j], ha='center', va='center',
                        color='r')
        cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
        plt.savefig(f'korelacje_obrazow/{oryginalne}_{i}_korelacja.png')
        plt.show()
        
