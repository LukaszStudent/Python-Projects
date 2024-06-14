import numpy as np
import random
import time

#zadanie 1
M=np.random.randint(0,100,(10,5))
C=np.trace(M)
print(M)
print(C)
D=np.diag(M)
print(D)

#zadanie 2
sigma1,sigma2, mu = 3,5,0.1
A=sigma1*np.random.randn(2,4)+4
B=sigma2*np.random.randn(2,4)+5
C=A*B
#print(A)
#print(B)
print(C)

#zadanie 3
A=np.random.randint(1,100,(1,5))
B=np.random.randint(1,100,(1,5))
C=A+B
#print(A)
#print(B)
print(C)


#zadanie 4
A=np.random.randint(0,100,(4,5))
B=np.random.randint(0,100,(5,4))
print(A)
B=np.transpose(B)
print(B)
print(A+B)


#zadanie 5
#print(A[:,2])
#print(B[:,3])
Q=A[:,2]*B[:,3]
print(Q)


#zadanie 6
A=np.random.normal(2,5,(4,6))
print(A)
odchy=np.std(A)
print(odchy)
sr=np.mean(A)
print(sr)
med=np.median(A)
print(med)
war=np.var(A)

B=np.random.uniform(-5,4,(4,6))
print(B)
odchy=np.std(B)
print(odchy)
sr=np.mean(B)
print(sr)
med=np.median(B)
print(med)
war=np.var(B)
print(war)


#zadanie 7
A=np.array([[1,2],[3,4]])
B=np.array([[5,6],[7,8]])
print(A)
print(B)
print(A*B)
C=np.dot(A,B)
print(C)
##Mnożenie przez "*" mnoży liczby na odpowiednich miejscach z jednej i drugiej macierzy 
##Mnożenie przy użyciu funkcji "dot" mnoży macierze tak jak się to robi standardowo/prawidlowo


#zadanie 8
A=np.arange(35).reshape((5,7))
print(A)
print(A.strides)
C=np.lib.stride_tricks.as_strided(A,(1,1,3,5),(0,0,28,4))#drugi argument(liczba_blokow_wierszy,liczba_blokow_kolumn,liczba_wierszy_w_bloku,liczba_kolumn_w_bloku)
                                                         #trzeci argument(od_ktorego_bitu_zaczyna_sie_nastepny_blok(kolumny),od_ktorego_bitu_zaczyna_sie_nastepny_blok(wiersze),w_jakiej_odleglosci_od_pierwszego_elem_w_bloku_zaczyna_sie_kolejny_wiersz,krok)
print(C)


#zadanie 9
A=np.zeros((2,2))
B=np.ones((2,2))
C=np.hstack((A,B,A))
print(C)
D=np.vstack((B,A))
print(D)
#hstack laczy macierze po horyzoncie, w sensie poziomo
#vstack laczy macierze pionowo
#warto to znac i uzywac gdy nie wiemy jak ma wygladac nasza koncowa macierz i tak jakby "doklejac" mniejsze kawalki do naszej macierzy


#zadanie 10
A = np.arange(24).reshape((4,6))
print(A)
print(A.strides)
C=np.lib.stride_tricks.as_strided(A,(2,2,2,3),(12,48,24,4))
print(C)
print(np.transpose(C.max((2,3))))
