#laboratorium 1
import matplotlib.pyplot as plt

tab=[]
for x in range(0,10,1):
    f= x**3 + x**2 + x
    tab.append(f)
print(tab)

plt.plot(tab)
plt.show()
