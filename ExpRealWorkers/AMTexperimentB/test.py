import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

'''x = np.arange(0, 10, 1)
y = -1 * np.exp(-0.4 * x + 0) * 19 + 6
plt.plot(y)



plt.ylim(0, 15)
plt.xlim(0, 15)
plt.show()'''

from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Example datasetspip
data1 = np.array([[1, 2], [3, 4], [5, 6]])
data2 = np.array([[10, 20], [30, 40], [50, 60]])

scaler = MinMaxScaler()

# Fit on the first dataset
scaler.fit(data1)

# Transform both datasets
data1_normalized = scaler.transform(data1)
data2_normalized = scaler.transform(data2)

print(data1_normalized, data2_normalized)