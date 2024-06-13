import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from PIL import Image 
import PIL 


informacja=cv2.imread('informacja.jpg')
oryginal=cv2.imread('0016.jpg')
auto=cv2.imread('0013.jpg')


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

def water_mark(img,mask,alpha=0.25):
    assert (img.shape[0]==mask.shape[0]) and (img.shape[1]==mask.shape[1]), "Wrong size"
    #assert (mask.dtype==bool), "Wrong type - mask"
    if len(img.shape)<3:
        flag=True
        t_img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGBA)
        t_mask=cv2.cvtColor((mask*255).astype(np.uint8),cv2.COLOR_GRAY2RGBA)
    else:
        flag=False
        t_img=cv2.cvtColor(img,cv2.COLOR_RGB2RGBA)       
        t_mask=cv2.cvtColor((mask).astype(np.uint8),cv2.COLOR_RGB2RGBA)
    t_out=cv2.addWeighted(t_img,1,t_mask,alpha,0)
    if flag:
        out=cv2.cvtColor(t_out,cv2.COLOR_RGBA2GRAY)
    else:
        out=cv2.cvtColor(t_out,cv2.COLOR_RGBA2RGB)
    return out


def put_data(img,data,binary_mask=np.uint8(1)):
    assert img.dtype==np.uint8 , "img wrong data type"
    assert binary_mask.dtype==np.uint8, "binary_mask wrong data type"
    un_binary_mask=np.unpackbits(binary_mask)
    if data.dtype!=bool:
        unpacked_data=np.unpackbits(data)
    else:
        unpacked_data=data
    dataspace=img.shape[0]*img.shape[1]*np.sum(un_binary_mask)
    assert (dataspace>=unpacked_data.size) , "too much data"
    if dataspace==unpacked_data.size:
        prepered_data=unpacked_data.reshape(img.shape[0],img.shape[1],np.sum(un_binary_mask)).astype(np.uint8)
    else:
        prepered_data=np.resize(unpacked_data,(img.shape[0],img.shape[1],np.sum(un_binary_mask))).astype(np.uint8)
    mask=np.full((img.shape[0],img.shape[1]),binary_mask)
    img=np.bitwise_and(img,np.invert(mask))
    bv=0
    for i,b in enumerate(un_binary_mask[::-1]):
        if b:
            temp=prepered_data[:,:,bv]
            temp=np.left_shift(temp,i)
            img=np.bitwise_or(img,temp)
            bv+=1
    return img

def pop_data(img,binary_mask=np.uint8(1),out_shape=None):
    un_binary_mask=np.unpackbits(binary_mask)
    data=np.zeros((img.shape[0],img.shape[1],np.sum(un_binary_mask))).astype(np.uint8)
    bv=0
    for i,b in enumerate(un_binary_mask[::-1]):
        if b:
            mask=np.full((img.shape[0],img.shape[1]),2**i)
            temp=np.bitwise_and(img,mask)           
            data[:,:,bv]=temp[:,:].astype(np.uint8)             
            bv+=1
    if out_shape!=None:
        tmp=np.packbits(data.flatten())        
        tmp=tmp[:np.prod(out_shape)]
        data=tmp.reshape(out_shape)
    return data


alpha=0.05
informacja=cv2.cvtColor(informacja,cv2.COLOR_RGB2GRAY)
mask=informacja
bit_red=1
bit_green=5
bit_blue=1

red=0
znak_wodny=water_mark(oryginal,mask,alpha=alpha)
kodowanie_red=put_data(oryginal[:,:,red],znak_wodny[:,:,red].astype(bool),binary_mask=np.uint8(bit_red))
dekodowanie_red=pop_data(kodowanie_red,binary_mask=np.uint8(bit_red))

green=1
kodowanie_green=put_data(oryginal[:,:,green],znak_wodny[:,:,green].astype(bool),binary_mask=np.uint8(bit_green))
dekodowanie_green=pop_data(kodowanie_green,binary_mask=np.uint8(bit_green))
popped_img = cv2.threshold(dekodowanie_green, 1, 255, cv2.THRESH_BINARY)

blue=2
kodowanie_blue=put_data(oryginal[:,:,blue],znak_wodny[:,:,blue].astype(bool),binary_mask=np.uint8(bit_blue))
dekodowanie_blue=pop_data(kodowanie_blue,binary_mask=np.uint8(bit_blue))

oryginal_gray=cv2.cvtColor(oryginal,cv2.COLOR_RGB2GRAY)
znak_wodny_gray=mask#cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
kodowanie_gray=put_data(oryginal_gray,znak_wodny_gray.astype(bool),np.uint8(1))
dekodowanie_gray=pop_data(kodowanie_gray)

nowy=np.dstack((oryginal[:,:,0],oryginal[:,:,1],kodowanie_blue))
obj=Miary(oryginal,znak_wodny)
mse=obj.MSE()
nmse=obj.NMSE()
psnr=obj.PSNR()
IF=obj.IF()
ssim=obj.SSIM()
print(f'Plik: Zakodowany obraz przy alpha={alpha}')
print(f'MSE: {mse}')
print(f'NMSE: {nmse}')  
print(f'PSNR: {psnr}')
print(f'IF: {IF}')
print(f'SSIM: {ssim}')
print()
print()





bits=64
f=open('text.txt','r')
text = f.read()
text = bytes(text,'utf-8')
text = np.frombuffer(text,dtype=np.uint8)
f.close() 
potegi=[1,2,4,8,16,32,64,128]
warstwy=[0,1,2]
for i in potegi:
    for j in warstwy:
        kodowany_text=put_data(oryginal[:,:,j],text,binary_mask=np.uint8(i))
        dekodowanie_text=pop_data(kodowany_text, binary_mask=np.uint8(i), out_shape=text.shape)
        if(j==0):
            kodowany_text=np.dstack([kodowany_text,oryginal[:,:,1],oryginal[:,:,2]])
        elif(j==1):
            kodowany_text=np.dstack([oryginal[:,:,0],kodowany_text,oryginal[:,:,2]])
        else:
            kodowany_text=np.dstack([oryginal[:,:,0],oryginal[:,:,1],kodowany_text])
        cv2.imshow('dsad',kodowany_text)
        cv2.waitKey(0)
        cv2.imwrite(f'text_bit{i}_warstwa{j}.jpg',kodowany_text)

        
#         # print('Zakodowany teks',dekodowanie_text)
#         obrazek=cv2.imread(f'text_bit{i}_warstwa{j}.jpg')
#         obj=Miary(oryginal,obrazek)
#         mse=obj.MSE()
#         nmse=obj.NMSE()
#         psnr=obj.PSNR()
#         IF=obj.IF()
#         ssim=obj.SSIM()
#         print(f'Plik: text_bit{i}_warstwa{j}.jpg')
#         print(f'MSE: {mse}')
#         print(f'NMSE: {nmse}')  
#         print(f'PSNR: {psnr}')  
#         print(f'IF: {IF}')
#         print(f'SSIM: {ssim}')
#         print()
#         print()

plt.subplot(3,3,1)
plt.title('Oryginal')
plt.xticks([])
plt.imshow(cv2.cvtColor(oryginal,cv2.COLOR_BGR2RGB))
plt.subplot(3,3,2)
plt.xticks([])
plt.yticks([])
plt.title('Information Coded\non LSB Blue')
polaczone=np.dstack((oryginal[:,:,0],oryginal[:,:,1],kodowanie_blue))
plt.imshow(cv2.cvtColor(polaczone,cv2.COLOR_BGR2RGB))
plt.subplot(3,3,3)
plt.xticks([])
plt.yticks([])
plt.title('Decoded information\nfrom LSB Blue')
zdekodowane_blue=cv2.cvtColor(dekodowanie_blue.astype(np.uint8),cv2.COLOR_BGR2RGB)
plt.imshow(dekodowanie_blue,cmap=plt.cm.gray)

plt.subplot(3,3,4)
plt.xticks([])
plt.title('Oryginal - Gray')
plt.imshow(oryginal_gray,cmap=plt.cm.gray)
plt.subplot(3,3,5)
plt.xticks([])
plt.yticks([])
plt.title('Information Coded\non LSB Grayscale')
plt.imshow(kodowanie_gray,cmap=plt.cm.gray)
plt.subplot(3,3,6)
plt.xticks([])
plt.yticks([])
plt.title('Information Coded\non 5 bit of Green')
zakodowane_green=np.dstack((kodowanie_red,kodowanie_green,kodowanie_blue))
plt.imshow(cv2.cvtColor(zakodowane_green,cv2.COLOR_BGR2RGB))

plt.subplot(3,3,7)
plt.title('Information')
plt.imshow(cv2.cvtColor(mask,cv2.COLOR_BGR2RGB))
plt.subplot(3,3,8)
plt.yticks([])
plt.title('Watermark color')
#znak_wodny=np.dstack([znak_wodny_red,znak_wodny_green,znak_wodny_blue])
plt.imshow(cv2.cvtColor(znak_wodny,cv2.COLOR_BGR2RGB))
plt.subplot(3,3,9)
plt.yticks([])
plt.title('Watermark gray')
kolor=cv2.cvtColor(znak_wodny,cv2.COLOR_BGR2GRAY)
plt.imshow(kolor,cmap=plt.cm.gray)
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
plt.show()