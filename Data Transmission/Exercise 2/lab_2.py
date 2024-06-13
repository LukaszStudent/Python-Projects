import math as m
import matplotlib.pyplot as plt
import numpy as np


################################################################
#zadanie 1
#funkcja nr 6 z tabeli nr 1

#Tc=10 #czas trwania sygnalu
#fs=20000 #czestotliwosc probkowania fs>=2*fmax
#f=10000 #czestotliwosc
#Ts=1/fs #okres probkowania
#N=round(Tc*fs,0) #liczba probek na caly sygnal
#tab=[]
#czas=[]
#for n in range(0,N,1):
#    t=n*Ts
#    fun = ((m.exp(-t)*m.sin(m.pi*f*t))/(2.0001+m.cos(m.pi*t)))
#    tab.append(fun)
#    czas.append(t)

#plt.plot(czas,tab)
#plt.title('Wykres x(t) -(zadanie-1)')
#plt.show()


################################################################
##zadanie 2
##funkcja nr 4 z tabeli nr 2
Tc=10 #czas trwania sygnalu
fs=20000 #czestotliwosc probkowania fs>=2*fmax
f=10000 #czestotliwosc
Ts=1/fs #okres probkowania
N=round(Tc*fs,0) #liczba probek na caly sygnal
tabx=[]
taby=[]
tabz=[]
tabv=[]
czas=[]
for n in range(0,N,1):
    t=n*Ts
    x = ((m.exp(-t)*m.sin(m.pi*f*t))/(2.0001+m.cos(m.pi*t)))
    tabx.append(x)
    y=-t**2*m.cos(t/0.2)*tabx[n]
    taby.append(y)
    z=tabx[n]*m.cos(2*m.pi*t**2+m.pi)+0.276*t**2*tabx[n]
    tabz.append(z)
    v=m.sqrt(abs(1.77-taby[n]+tabz[n])*m.cos(5.2*m.pi*t)+tabx[n]+4)
    tabv.append(v)
    czas.append(t)


plt.subplot(2,2,1)
plt.plot(czas,tabx)
plt.title('wykres x(t)')

plt.subplot(2,2,2)
plt.plot(czas,taby)
plt.title('wykres y(t)')

plt.subplot(2,2,3)
plt.plot(czas,tabz)
plt.title('wykres z(t)')

plt.subplot(2,2,4)
plt.plot(czas,tabv)
plt.title('wykres v(t)')

plt.show()


########################################################
#zadanie 3
#funkcja nr 3 z tabeli nr 3
#Tc=10 #czas trwania sygnalu
#fs=20000 #czestotliwosc probkowania fs>=2*fmax
#f=10000 #czestotliwosc
#Ts=1/fs #okres probkowania
#N=round(Tc*fs,0) #liczba probek na caly sygnal
#czas=[]
#tabu=[]
#for n in range(0,N,1):
#    t=n*Ts
#    if t>=0 and t<1.2:
#        u=(-t**2+0.5)*m.sin(30*m.pi*t)*m.log2(t**2+1)
#        tabu.append(u)
#        czas.append(t)
#    if t>=1.2 and t<2:
#        u=(1/t)*0.8*m.sin(25*m.pi*t)-0.1*t
#        tabu.append(u)
#        czas.append(t)
#    if t>=2 and t<2.4:
#        u=abs(m.sin(2*m.pi*t**2))**0.8
#        tabu.append(u)
#        czas.append(t)
#    if t>=2.4 and t<3.1:
#        u=0.23*m.sin(20*m.pi*t)*m.sin(12*m.pi*t)
#        tabu.append(u)
#        czas.append(t)

#plt.plot(czas,tabu)
#plt.title('Wykres u(t) -(zadanie-3)')
#plt.show()

########################################################
#zadanie 4
#funkcja nr 14 z tabeli nr 4

fs=22050
Tc=1
N=round(Tc*fs,0) #liczba probek na caly sygnal
#f=11025 #czestotliwosc
Ts=1/fs #okres probkowania
lista1,lista2,lista3=[],[],[]
czas1,czas2,czas3=[],[],[]
suma1,suma2,suma3=0,0,0
for n in range(0,N,1):
    t=n*Ts
    for h in range(1,3,1):
        b=(m.sin((h*t*m.pi)/2+m.cos(h**2*m.pi*t)))
        suma1+=b
        lista1.append(suma1)
        czas1.append(t)

    for h in range(1,7,1):
        b=(m.sin((h*t*m.pi)/2+m.cos(h**2*m.pi*t)))
        suma2+=b
        lista2.append(suma2)
        czas2.append(t)

    for h in range(1,11,1):
        b=(m.sin((h*t*m.pi)/2+m.cos(h**2*m.pi*t)))
        suma3+=b
        lista3.append(suma3)
        czas3.append(t)

plt.subplot(1,3,1)
plt.plot(czas1,lista1)
plt.title('$b_1(t)$')

plt.subplot(1,3,2)
plt.plot(czas2,lista2)
plt.title('$b_2(t)$')

plt.subplot(1,3,3)
plt.plot(czas3,lista3)
plt.title('$b_3(t)$')

plt.show()
