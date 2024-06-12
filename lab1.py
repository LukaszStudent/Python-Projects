import numpy as np
import matplotlib.pyplot as plt
import cv2

oryginal=plt.imread('lena.jpg')
img2=plt.imread('pomidorek.png')
# print(img1.dtype)#uint8
# print(img1.shape)

#print(img2.dtype)#float32
#print(img2.shape)

# plt.imshow(img1)
# plt.show()

#plt.imshow(img2)
#plt.show()

#img1=cv2.imread('pic1.jpg')
#print(img1.dtype)
#print(img1.shape)


#img2=cv2.imread('pic2.png')
#print(img2.dtype)
#print(img2.shape)

red=oryginal[:,:,0]
green=oryginal[:,:,1]
blue=oryginal[:,:,2]
plt.imshow(red)
plt.title('czerwony')
print(red.shape)
plt.show()
plt.imshow(green)
plt.title('zielony')
plt.show()
plt.imshow(blue)
plt.title('niebieski')
plt.show()

plt.imshow(red,cmap=plt.cm.gray)#, vmin=0, vmax=120)
plt.show()

Y1=0.299*red + 0.587*green + 0.114*blue
Y2=0.2126*red+ 0.7152*green + 0.0722*blue


#zadanie 1
plt.subplot(3,3,1)
plt.imshow(oryginal)

plt.subplot(3,3,2)
plt.imshow(Y1)

plt.subplot(3,3,3)
plt.imshow(Y2)

plt.subplot(3,3,4)
plt.imshow(red)

plt.subplot(3,3,5)
plt.imshow(green)

plt.subplot(3,3,6)
plt.imshow(blue)

# plt.subplot(3,3,7)
# plt.imshow(R)

# plt.subplot(3,3,8)
# plt.imshow(G)

# plt.subplot(3,3,9)
# plt.imshow(B)

plt.show()