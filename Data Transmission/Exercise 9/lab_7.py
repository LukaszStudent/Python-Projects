import math as m

class Hamming_7_4():
    def __init__(self,slowo):
        self.slowo=slowo
        if len(slowo)>4:
            self.slowo=self.slowo[:4]

    def koder(self):
        global zakodowane
        print(self.slowo)
        self.slowo.insert(0,5)
        self.slowo.insert(1,5)
        self.slowo.insert(3,5)
        self.slowo[0]=((self.slowo[2]%2)+(self.slowo[4]%2)+(self.slowo[6]%2))%2
        self.slowo[1]=((self.slowo[2]%2)+(self.slowo[5]%2)+(self.slowo[6]%2))%2
        self.slowo[3]=((self.slowo[4]%2)+(self.slowo[5]%2)+(self.slowo[6]%2))%2
        print('slowo',self.slowo)
        zakodowane=self.slowo

    def dekoder(self,slowo):
        slowo[4]=0
        print('slowo zmienione',slowo)

        x1_Prim=((slowo[2]%2)+(slowo[4]%2)+(slowo[6]%2))%2
        x2_Prim=((slowo[2]%2)+(slowo[5]%2)+(slowo[6]%2))%2
        x4_Prim=((slowo[4]%2)+(slowo[5]%2)+(slowo[6]%2))%2

        x1_Daszek=((slowo[0]%2)+(x1_Prim%2))%2
        x2_Daszek=((slowo[1]%2)+(x2_Prim%2))%2
        x4_Daszek=((slowo[3]%2)+(x4_Prim%2))%2

        S=x1_Daszek*2**0+x2_Daszek*2**1+x4_Daszek*2**2
        print('Wykryty blad na',S,'pozycji w sygnale')
        

#lista=[1,1,0,1,0,0,1]

#k=Hamming_7_4(lista)
#k.koder()
#k.dekoder()


       
