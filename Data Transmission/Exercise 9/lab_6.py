import lab_5 as lab5
import math as m
import matplotlib.pyplot as plt
from statistics import mode


class Demodulator_ASK():
    def __init__(self,sygnal_ASK,A,Tc,fs,f,fm,W):
        self.sygnal_ASK=sygnal_ASK
        self.A=A
        self.Tc=Tc
        self.B=len(self.sygnal_ASK)
        self.Tb=self.Tc/self.B
        self.fs=fs
        self.f=f
        self.fm=fm
        self.Ts=1/self.fs
        self.N=round(self.Tc*self.fs,0)
        
        self.W=W
        self.fn=self.W*(1/self.Tb)
        self.liczba_probek_na_jeden_sygnal=int(self.Tb*self.fs)


    def x_t(self):
        global lxt
        czasD_ASK=[]
        lxt=[]
        start=0
        stop=self.liczba_probek_na_jeden_sygnal
        for n in range(0,self.B,1):
            for i in range(start,stop,1):
                t=i*self.Ts
                czasD_ASK.append(t)
                xt=self.sygnal_ASK[i]*(self.A*(m.sin(2*m.pi*self.fn*t)))
                lxt.append(xt)
            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        #plt.title('x(t)')
        plt.plot(czasD_ASK,lxt,label='x(t)')
        plt.legend()
        #plt.show()


    def p_t(self):
        global lpt
        czasD_ASK=[]
        lpt=[]
        start=0
        stop=self.liczba_probek_na_jeden_sygnal
        for n in range(0,self.B,1):
            pt=0
            for i in range(start,stop,1):
                t=i*self.Ts
                czasD_ASK.append(t)
                pt+=lxt[i]
                lpt.append(pt)
            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        #plt.title('p(t)')
        plt.plot(czasD_ASK,lpt,label='p(t)')
        plt.legend()
        #plt.show()

    def c_t(self,h):
        global wartosc_DASK_ct
        czasD_ASK=[]
        lct=[]
        wartosc_DASK_ct=[]

        start=0
        stop=self.liczba_probek_na_jeden_sygnal
        for n in range(0,self.B,1):
            pt=0
            for i in range(start,stop,1):
                t=i*self.Ts
                czasD_ASK.append(t)
                if lpt[i]>h:
                    wartosc_DASK_ct.append(1)
                else:
                    wartosc_DASK_ct.append(0)

            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        #plt.title('c(t)')
        plt.plot(czasD_ASK,wartosc_DASK_ct,label='c(t)')
        plt.legend()
        #plt.show()

    def konwersja(self):
            start=0
            stop=self.liczba_probek_na_jeden_sygnal
            czas=[]
            wartosc=[]
            bit_wartosc=[]
            jeden_bit=[]

            for i in range(0,self.B,1):
                for j in range(start,stop,1):
                    if wartosc_DASK_ct[j]==1:
                        jeden_bit.append(1)
                    else:
                        jeden_bit.append(0)
                bit_wartosc.append(mode(jeden_bit))
                jeden_bit.clear()
                print(bit_wartosc)
                start=stop
                stop=stop+self.liczba_probek_na_jeden_sygnal
                if stop>int(self.N):
                    stop=int(self.N)

            start=0
            stop=self.liczba_probek_na_jeden_sygnal
            for i in range(0,self.B,1):
                for j in range(start,stop,1):
                    t=j*self.Ts
                    czas.append(t)
                    if bit_wartosc[i]==1:
                        jeden_bit.append(1)
                    else:
                        jeden_bit.append(0)
                start=stop
                stop=stop+self.liczba_probek_na_jeden_sygnal
                if stop>int(self.N):
                    stop=int(self.N)

            plt.plot(czas,jeden_bit,label='zamieniony na bity')
            plt.legend()
            #plt.show()


class Demodulator_PSK(Demodulator_ASK):
    def c_t(self):
        global wartosc_DPSK_ct
        czasD_PSK=[]
        lct=[]
        wartosc_DPSK_ct=[]

        start=0
        stop=self.liczba_probek_na_jeden_sygnal
        for n in range(0,self.B,1):
            pt=0
            for i in range(start,stop,1):
                t=i*self.Ts
                czasD_PSK.append(t)
                if lpt[i]<0:
                    wartosc_DPSK_ct.append(1)
                else:
                    wartosc_DPSK_ct.append(0)

            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        #plt.title('c(t)')
        plt.plot(czasD_PSK,wartosc_DPSK_ct,label='c(t)')
        plt.legend()
        #plt.show()
        return wartosc_DPSK_ct

    def konwersja(self):
        start=0
        stop=self.liczba_probek_na_jeden_sygnal
        czas=[]
        wartosc=[]
        bit_wartosc=[]
        jeden_bit=[]

        for i in range(0,self.B,1):
            for j in range(start,stop,1):
                if wartosc_DPSK_ct[j]==1:
                    jeden_bit.append(1)
                else:
                    jeden_bit.append(0)
            bit_wartosc.append(mode(jeden_bit))
            jeden_bit.clear()
            print(bit_wartosc)
            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        start=0
        stop=self.liczba_probek_na_jeden_sygnal
        for i in range(0,self.B,1):
            for j in range(start,stop,1):
                t=j*self.Ts
                czas.append(t)
                if bit_wartosc[i]==1:
                    jeden_bit.append(1)
                else:
                    jeden_bit.append(0)
            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        plt.plot(czas,jeden_bit,label='zamieniony na bity')
        plt.legend()
        #plt.show()



class Demodulator_FSK():
    def __init__(self,sygnal_FSK,A,Tc,fs,f,fm,W):
        self.sygnal_FSK=sygnal_FSK
        self.A=A
        self.Tc=Tc
        self.B=len(self.sygnal_FSK)
        self.Tb=self.Tc/self.B
        self.fs=fs
        self.f=f
        self.fm=fm
        self.Ts=1/self.fs
        self.N=round(self.Tc*self.fs,0)
        
        self.W=W
        self.fn=self.W*(1/self.Tb)
        self.liczba_probek_na_jeden_sygnal=int(self.Tb*self.fs)
        self.fn1=(self.W+1)/self.Tb
        self.fn2=(self.W+2)/self.Tb

    def x1_t(self):
        global lx1t
        czasD_FSK=[]
        lx1t=[]
        start=0
        stop=self.liczba_probek_na_jeden_sygnal
        for n in range(0,self.B,1):
            for i in range(start,stop,1):
                t=i*self.Ts
                czasD_FSK.append(t)
                xt=self.sygnal_FSK[i]*(self.A*(m.sin(2*m.pi*self.fn1*t)))
                lx1t.append(xt)
            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        #plt.title('x1(t)')
        plt.plot(czasD_FSK,lx1t,label='x1(t)')
        plt.legend()
        #plt.show()


    def x2_t(self):
        global lx2t
        czasD_FSK=[]
        lx2t=[]
        start=0
        stop=self.liczba_probek_na_jeden_sygnal
        for n in range(0,self.B,1):
            for i in range(start,stop,1):
                t=i*self.Ts
                czasD_FSK.append(t)
                xt=self.sygnal_FSK[i]*(self.A*(m.sin(2*m.pi*self.fn2*t)))
                lx2t.append(xt)
            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        #plt.title('x2(t)')
        plt.plot(czasD_FSK,lx2t,label='x2(t)')
        plt.legend()
        #plt.show()

    def p1_t(self):
        global lp1t
        czasD_FSK=[]
        lp1t=[]
        start=0
        stop=self.liczba_probek_na_jeden_sygnal
        for n in range(0,self.B,1):
            pt=0
            for i in range(start,stop,1):
                t=i*self.Ts
                czasD_FSK.append(t)
                pt+=lx1t[i]
                lp1t.append(pt)
            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        #plt.title('p1(t)')
        plt.plot(czasD_FSK,lp1t,label='p1(t)')
        plt.legend()
        #plt.show()

    def p2_t(self):
        global lp2t
        czasD_FSK=[]
        lp2t=[]
        start=0
        stop=self.liczba_probek_na_jeden_sygnal
        for n in range(0,self.B,1):
            pt=0
            for i in range(start,stop,1):
                t=i*self.Ts
                czasD_FSK.append(t)
                pt+=lx2t[i]
                lp2t.append(pt)
            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        #plt.title('p2(t)')
        plt.plot(czasD_FSK,lp2t,label='p2(t)')
        plt.legend()
        #plt.show()

    def p3_t(self):
        global lpt3
        czasD_FSK=[]
        lpt3=[]
        start=0
        stop=self.liczba_probek_na_jeden_sygnal
        for n in range(0,self.B,1):
            #pt=0
            for i in range(start,stop,1):
                t=i*self.Ts
                czasD_FSK.append(t)
                pt3=(-1)*lp1t[i]+lp2t[i]
                lpt3.append(pt3)
            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        #plt.title('p(t)')
        plt.axhline(y=0,c='r')
        plt.plot(czasD_FSK,lpt3,label='p(t)')
        plt.legend()
        #plt.show()


    def c_t(self):
        global wartosc_DFSK_ct
        czasD_FSK=[]
        lct=[]
        wartosc_DFSK_ct=[]

        start=0
        stop=self.liczba_probek_na_jeden_sygnal
        for n in range(0,self.B,1):
            pt3=0
            for i in range(start,stop,1):
                t=i*self.Ts
                czasD_FSK.append(t)
                if lpt3[i]>0:
                    wartosc_DFSK_ct.append(1)
                else:
                    wartosc_DFSK_ct.append(0)

            start=stop
            stop=stop+self.liczba_probek_na_jeden_sygnal
            if stop>int(self.N):
                stop=int(self.N)

        #plt.title('c(t)')
        plt.plot(czasD_FSK,wartosc_DFSK_ct,label='c(t)')
        plt.legend()
        #plt.show()

        return wartosc_DFSK_ct


def konwersja(sygnal,liczba_probek_na_jeden_sygnal,B,N,Ts):
        start=0
        stop=liczba_probek_na_jeden_sygnal
        czas=[]
        wartosc=[]
        bit_wartosc=[]
        jeden_bit=[]

        for i in range(0,B,1):
            for j in range(start,stop,1):
                if sygnal[j]==1:
                    jeden_bit.append(1)
                else:
                    jeden_bit.append(0)
            bit_wartosc.append(mode(jeden_bit))
            jeden_bit.clear()
            #print(bit_wartosc)
            start=stop
            stop=stop+liczba_probek_na_jeden_sygnal
            if stop>int(N):
                stop=int(N)

        start=0
        stop=liczba_probek_na_jeden_sygnal
        for i in range(0,B,1):
            for j in range(start,stop,1):
                t=j*Ts
                czas.append(t)
                if bit_wartosc[i]==1:
                    jeden_bit.append(1)
                else:
                    jeden_bit.append(0)
            start=stop
            stop=stop+liczba_probek_na_jeden_sygnal
            if stop>int(N):
                stop=int(N)

        plt.plot(czas,jeden_bit,label='zamieniony na bity')
        plt.legend()
        #plt.show()


#print('Demodulator ASK\n')
#plt.subplot(5,1,1)
#plt.title('Demodulator ASK')
#lab5.ASK(12,33,lab5.b)

#DASK=Demodulator_ASK(lab5.lask)
#plt.subplot(5,1,2)
#DASK.x_t(12)
#plt.subplot(5,1,3)
#DASK.p_t()
#plt.subplot(5,1,4)
#DASK.c_t(300000)
#plt.subplot(5,1,5)
#konwersja(wartosc_DASK_ct)
#plt.show()


#print('Demodulator PSK\n')
#plt.subplot(5,1,1)
#plt.title('Demodulator PSK')
#lab5.PSK(lab5.b)

#DPSK=Demodulator_PSK(lab5.lpsk)
#plt.subplot(5,1,2)
#DPSK.x_t(12)
#plt.subplot(5,1,3)
#DPSK.p_t()
#plt.subplot(5,1,4)
#DPSK.c_t()
#plt.subplot(5,1,5)
#konwersja(wartosc_DPSK_ct)
#plt.show()

##konwersja(DPSK.c_t())

#print('Demodulator FSK\n')
#plt.subplot(5,2,1)
#lab5.FSK(lab5.b)
#plt.title('Demodulator FSK')

#DFSK=Demodulator_FSK(lab5.lfsk)
#plt.subplot(5,2,3)
#DFSK.x1_t(12)
#plt.subplot(5,2,4)
#DFSK.x2_t(24)
#plt.subplot(5,2,5)
#DFSK.p1_t()
#plt.subplot(5,2,6)
#DFSK.p2_t()
#plt.subplot(5,2,7)
#DFSK.p3_t()
#plt.subplot(5,2,8)
#DFSK.c_t()
#plt.subplot(5,2,9)
#konwersja(wartosc_DFSK_ct)
#plt.show()
