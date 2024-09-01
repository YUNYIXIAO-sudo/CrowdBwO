#time-batch-multi-objective-ucb
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt
from AMTexperimentA.fee import feePara
#from DesignBatch import designBatch

nArms = 2
batchCreateTime = 1


'''20% fee on the reward and bonus amount (if any) you pay Workers. 
Tasks with 10 or more assignments will be charged an additional 20% fee 
on the reward you pay Workers. 
The minimum fee is $0.01 per assignment or bonus payment. --amt'''

def feeCount(Workers):
    fee = Workers * feePara

    fee[Workers == 0] = 0
    
    return fee


class estimateTraffic():
    def getTraffic(remainTime, paraMean, paraStd):
        test = 1
        t = 2
        mtime = np.zeros(nArms)
        traffic = np.zeros(nArms)
        batchTraffic = np.zeros(nArms)
        batchTrafficR = estimateTraffic.predict(1, paraMean, paraStd)
        

        while test == 1:
            if np.all(mtime != 0) == 0:
                batchTraffic = estimateTraffic.predict(t, paraMean, paraStd)
                

                get = np.where(batchTrafficR / (t - 1) > batchTraffic / t)[0]

                set = np.where(mtime == 0)[0]

                set = np.intersect1d(get, set)

                for s in set:
                    mtime[s] = t - 1
                    traffic[s] = batchTrafficR[s]
                   
                    
                if t <= remainTime:
                    t += 1
                    batchTrafficR = batchTraffic.copy()
                else:
                    traffic = np.where(mtime == 0, batchTrafficR, traffic)
                    mtime = np.where(mtime == 0, remainTime, mtime)
                    test = 0

            else:
                test = 0

        

        newTrafficPara = traffic / (mtime)
        newFeePara = feeCount(traffic) / (mtime)
        
        return newTrafficPara, newFeePara, mtime

    

    def predict(batchTime, paraMean, paraStd):
        bct = np.ones(nArms) * batchCreateTime
        batchTime -= bct
        batchTime = np.where(batchTime <= 0, 0.5, batchTime)

        result = -1 * np.exp(-1 * batchTime * paraMean[0] + paraMean[1]) * paraMean[2] + paraMean[3]
        result = np.where(batchTime < 1, 0, result)
        result = np.where(result < 0, 0, result)

        return result
    

    
'''def getAverageTraffic(traffic):

        newTrafficPara = (traffic / len(traffic[0])).copy
        newFeePara = (feeCount(traffic) / len(traffic[0])).copy()

        return newTrafficPara, newFeePara'''


#tPara, fPara, mtime = estimateTraffic.getTraffic(100, np.array([[1, 1], [1, 1], [1, 1]]), np.array([[0, 0],[0, 0],[0, 0]]))
