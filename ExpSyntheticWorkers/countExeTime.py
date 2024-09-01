import pandas as pd
import numpy as np

dataLP = pd.read_csv('exeTimeLP.csv')
dataMO = pd.read_csv('exeTimeMO.csv')


timeLPG0 = np.transpose(np.array(dataLP))[1][:10]
timeLPG1 = np.transpose(np.array(dataLP))[1][10:20]

trialLPG0 = np.transpose(np.array(dataLP))[7][:10]
trialLPG1 = np.transpose(np.array(dataLP))[7][10:20]


timeMOG0 = np.transpose(np.array(dataMO))[1][:10]
timeMOG1 = np.transpose(np.array(dataMO))[1][10:20]

trialMOG0 = np.transpose(np.array(dataMO))[7][:10]
trialMOG1 = np.transpose(np.array(dataMO))[7][10:20]

print(timeLPG0 / trialLPG0)
print(np.mean(timeLPG0 / trialLPG0, axis=0))
print(np.mean(timeLPG1 / trialLPG1, axis=0))

print(np.mean(timeMOG0 / trialMOG0, axis=0))
print(np.mean(timeMOG1 / trialMOG1, axis=0))