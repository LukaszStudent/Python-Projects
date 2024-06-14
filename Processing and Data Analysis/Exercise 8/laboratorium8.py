import pandas as pd
import matplotlib.pyplot as plt

dane=pd.read_csv('autos.csv')
print(dane)

print(dane.columns)

plt.scatter(dane['length'],dane['make'])
plt.show()
