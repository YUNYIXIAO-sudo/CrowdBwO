import numpy as np
from matplotlib import pyplot as plt
from parameter import para


class designBatch():
    def arithmetic(time, remainTime, pullingArms):
        batchTime = np.max(time * pullingArms)
        
        if batchTime > remainTime:
            return remainTime
        elif batchTime <= remainTime:
            return batchTime
        
    def arithmeticNoTimeLimit(time, pullingArms):
        batchTime = np.max(time * pullingArms)
        return batchTime
        

'''class wTH():
    def predict(pulled, time):
        time = np.tile(time, para.nArms)
        result = -np.exp(-0.6 * time + 5) + 0 * pulled + 100 

        return result
    
class wTraffic():
    def predict(time):
        result = np.exp(-1 * (time - env.para.batchCreateTime) + 5) + 0 * (100) + 5

        result = np.where(env.para.batchCreateTime >= time, 0, result)

        return result
    
    



dB = designBatch()
dB.tune([0, 1, 1], wTraffic)

print(wT.complexPredict(100, 10))'''