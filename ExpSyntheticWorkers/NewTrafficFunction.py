#time-batch-multi-objective-ucb
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt
#from DesignBatch import designBatch
from Parameters import para
from environment import env, feeCountSeperate


class estimateTraffic():
    def getTraffic(remainTime, paraMean, paraStd, experiment, workerset):
        test = 1
        t = 2
        mtime = np.zeros(para.nArms[workerset])
        traffic = np.zeros(para.nArms[workerset])
        batchTraffic = np.zeros(para.nArms[workerset])
        batchTrafficR = estimateTraffic.predict(1, paraMean, paraStd, experiment, workerset)
        

        while test == 1:
            if np.all(mtime != 0) == 0:
                batchTraffic = estimateTraffic.predict(t, paraMean, paraStd, experiment, workerset)

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
        newFeePara = feeCountSeperate(traffic, experiment, workerset) / (mtime)
        return newTrafficPara, newFeePara, mtime


    def getTrafficNoTimeLimit(paraMean, paraStd, experiment, workerset):
        test = 1
        t = 2
        mtime = np.zeros(para.nArms[workerset])
        traffic = np.zeros(para.nArms[workerset])
        batchTraffic = np.zeros(para.nArms[workerset])
        batchTrafficR = estimateTraffic.predict(1, paraMean, paraStd, experiment, workerset)

        while test == 1:
            if np.all(mtime != 0) == 0:
                batchTraffic = estimateTraffic.predict(t, paraMean, paraStd, experiment, workerset)

                get = np.where(batchTrafficR / (t - 1) > (batchTraffic) / t)[0]

                set = np.where(mtime == 0)[0]

                set = np.intersect1d(get, set)

                for s in set:
                    mtime[s] = t - 1
                    traffic[s] = batchTrafficR[s]
                    

                if t <= 1000:
                    
                    t += 1
                    batchTrafficR = batchTraffic.copy()
                    
                else:
                    traffic = np.where(mtime == 0, batchTrafficR, traffic)
                    mtime = np.where(mtime == 0, 1000, mtime)
                    
                    test = 0

            else:
                test = 0

        
        newTrafficPara = traffic / (mtime)
        newFeePara = feeCountSeperate(traffic, experiment, workerset) / (mtime)

        
        return newTrafficPara, newFeePara, mtime


    def getDensityTraffic(remainTime, paraMean, paraStd, experiment, workerset):
        test = 1
        t = 2
        mtime = np.zeros(para.nArms[workerset])
        traffic = np.zeros(para.nArms[workerset])
        batchTraffic = np.zeros(para.nArms[workerset])
        batchTrafficR = estimateTraffic.predict(1, paraMean, paraStd, experiment, workerset)
        feeR = feeCountSeperate(batchTrafficR, experiment, workerset)

        while test == 1:
            if np.all(mtime != 0) == 0:
                batchTraffic = estimateTraffic.predict(t, paraMean, paraStd, experiment, workerset)
                fee = feeCountSeperate(batchTrafficR + batchTraffic, experiment, workerset)

    
                if np.all(fee > 0) == 1 and np.all(feeR > 0) == 1:
                    get = np.where(batchTrafficR / feeR > (batchTraffic) / fee)[0]
                    set = np.where(mtime == 0)[0]

                    set = np.intersect1d(get, set)

                    for s in set:
                        mtime[s] = t - 1
                        traffic[s] = batchTrafficR[s]
                        
                    

                if t <= remainTime:
                    t += 1
                    batchTrafficR = batchTraffic.copy()
                    feeR = fee
                else:
                    traffic = np.where(mtime == 0, batchTrafficR, traffic)
                    mtime = np.where(mtime == 0, remainTime, mtime)
                    
                    test = 0

            else:
                test = 0
        
        newTrafficPara = traffic / (mtime)
        newFeePara = feeCountSeperate(traffic, experiment, workerset) / (mtime)

        
        return newTrafficPara, newFeePara, mtime
    

    def getDensityTrafficNoTimeLimit(paraMean, paraStd, experiment, workerset):
        test = 1
        t = 2
        mtime = np.zeros(para.nArms[workerset])
        traffic = np.zeros(para.nArms[workerset])
        batchTraffic = np.zeros(para.nArms[workerset])
        batchTrafficR = estimateTraffic.predict(1, paraMean, paraStd, experiment, workerset)
        feeR = feeCountSeperate(batchTrafficR, experiment, workerset)

        while test == 1:
            if np.all(mtime != 0) == 0:
                batchTraffic = estimateTraffic.predict(t, paraMean, paraStd, experiment, workerset)
                fee = feeCountSeperate(batchTrafficR + batchTraffic, experiment, workerset)

    
                if np.all(fee > 0) == 1 and np.all(feeR > 0) == 1:
                    get = np.where(batchTrafficR / feeR > (batchTraffic) / fee)[0]
                    set = np.where(mtime == 0)[0]

                    set = np.intersect1d(get, set)

                    for s in set:
                        mtime[s] = t - 1
                        traffic[s] = batchTrafficR[s]
                    

                if t <= 1000:
                    t += 1
                    batchTrafficR = batchTraffic.copy()
                    feeR = fee
                else:
                    traffic = np.where(mtime == 0, batchTrafficR, traffic)
                    mtime = np.where(mtime == 0, 1000, mtime)
                    
                    test = 0

            else:
                test = 0
        
        newTrafficPara = traffic / (mtime)
        newFeePara = feeCountSeperate(traffic, experiment, workerset) / (mtime)

        
        return newTrafficPara, newFeePara, mtime
    

    def predict(batchTime, paraMean, paraStd, experiment, workerset):
        batchTime -= np.array(para.batchCreateTime[workerset], dtype=np.int32)
        batchTime = np.where(batchTime <= 0, 0.5, batchTime)

        result = -1 * np.exp(-1 * batchTime * paraMean[0] + paraMean[1]) * paraMean[2] + paraMean[3]
        result = np.where(batchTime < 1, 0, result)
        result = np.where(result < 0, 0, result)

        return result
    
