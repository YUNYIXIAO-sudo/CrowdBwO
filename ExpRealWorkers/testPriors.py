import numpy as np
import pandas as pd
from WorkerTraffic import workerTraffic
from matplotlib import pyplot as plt


dfSPLP = pd.read_csv('AMTexperimentB/exp03-21/realEXPresultsSPMO21.csv')


rewardSPLP1 = np.array(dfSPLP['0'])
pulledSPLP1 = np.array(dfSPLP['2'])
usedBudgetSPLP1 = np.array(dfSPLP['4'])
FeasibleSPLP1 = np.array(dfSPLP['7'])
usedTimeSPLP1 = np.sum(FeasibleSPLP1)

rewardSPLP2 = np.array(dfSPLP['1'])
pulledSPLP2 = np.array(dfSPLP['3'])
usedBudgetSPLP2 = np.array(dfSPLP['5'])
FeasibleSPLP2 = np.array(dfSPLP['8'])
usedTimeSPLP2 = np.sum(FeasibleSPLP2)


timeSPLP = np.array(dfSPLP['6'])

#use the data to train the worker throughput Models for two worker sets

meansSPLP1, stdsSPLP1 = workerTraffic.train((np.array(timeSPLP) * np.array(FeasibleSPLP1))[:100], (np.array(pulledSPLP1) * np.array(FeasibleSPLP1))[100], 0)
meansSPLP2, stdsSPLP2 = workerTraffic.train((np.array(timeSPLP) * np.array(FeasibleSPLP2))[:int(usedTimeSPLP2)+1], (np.array(pulledSPLP2) * np.array(FeasibleSPLP2))[:int(usedTimeSPLP2)+1], 1)
meansSPLP = np.column_stack((meansSPLP1, meansSPLP2))
stdsSPLP = np.column_stack((stdsSPLP1, stdsSPLP2))
print(meansSPLP, stdsSPLP, 'training results')


x = np.arange(0, 400, 1)
y1 = -np.exp(-meansSPLP1[0] * x + meansSPLP1[1]) * meansSPLP1[2] + meansSPLP1[3]
y2 = -np.exp(-meansSPLP2[0] * x + meansSPLP2[1]) * meansSPLP2[2] + meansSPLP2[3]

plt.plot(y1, color='r', label='Model1')
plt.plot(y2, color='b', label='Model2')
plt.plot(np.array(timeSPLP) * np.array(FeasibleSPLP1), np.array(pulledSPLP1) * np.array(FeasibleSPLP1), 'ro', label='Actual1')
plt.plot(np.array(timeSPLP) * np.array(FeasibleSPLP2), np.array(pulledSPLP2) * np.array(FeasibleSPLP2), 'bo', label='Actual2')
plt.legend(fontsize=15)
plt.show()


'''
[ 5.31479208e-02 -4.07586489e+00  2.99472707e+03  5.16597597e+01] [4.14574759e-03 6.31861818e-02 9.47583041e+00 1.59599526e+00]
[  0.21302715  -0.33907899 151.15251274  92.64361355] [0.03760001 0.09940782 9.63201901 5.94109368]'''
