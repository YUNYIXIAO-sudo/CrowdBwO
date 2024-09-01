from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 20})

x = np.arange(0, 400, 1)
y = -1 * np.exp(-0.07 * x -4) * 3000  + 30

#plt.plot(y)
#plt.show()

a = np.exp(10)
print(a)
