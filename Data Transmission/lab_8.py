import numpy as np 

class Hamming_15_11():
    def __init__(self, slowo):
        self.slowo=slowo
        if len(self.slowo)>11:
            self.slowo=self.slowo[:15]

    def koder(self):
        print('słowo kodowane',self.slowo)
        m=4
        k=11
        matrix_one=np.eye(k,dtype=int)

        binarne=[]
        for i in enumerate(self.slowo):
            idx=i[0]+1
            idx_bin=bin(idx)[2:]

            if len(idx_bin)==1:
               idx_bin='000'+str(idx_bin)
            if len(idx_bin)==2:
               idx_bin='00'+str(idx_bin)
            if len(idx_bin)==3:
               idx_bin='0'+str(idx_bin)

            binarne.append(idx_bin)

        x1,x2,x4,x8=[],[],[],[]
        for i in range(0,len(binarne),1):
            x1.append(int(binarne[i][-1]))
            x2.append(int(binarne[i][-2]))
            x4.append(int(binarne[i][-3]))
            x8.append(int(binarne[i][-4]))

        matrix_P=np.vstack((x1,x2,x4,x8)).T
        self.matrix_P=np.delete(matrix_P,(0,1,3,7),0)

        matrix_G=np.hstack((self.matrix_P,matrix_one))
        
        b=self.slowo
        b=np.delete(b,(0,1,3,7),0)
        vector_b=np.array(b).reshape((1,11))

        self.c=vector_b@matrix_G
        self.c=self.c%2
        self.slowo=self.c
        print('slowo zakodowane',self.slowo)

    def dekoder(self):
        self.slowo[0][4]=1
        print('slowo przeklamane',self.slowo)
        k=11
        n=15
        matrix_one=np.eye(n-k,dtype=int)
        matrix_H=np.hstack((matrix_one,self.matrix_P.T))
        s=(self.c@matrix_H.T)
        s=s%2

        S=s[0][0]*2**0+s[0][1]*2**1+s[0][2]*2**2+s[0][3]*2**3
        print('Blad na',S,'pozycji')



lista=[1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1]

H = Hamming_15_11(lista)
H.koder()
H.dekoder()