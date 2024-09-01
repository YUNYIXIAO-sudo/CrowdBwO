import numpy as np
from Parameters import para

np.random.seed(32)

class env():
    
    def reactRewards(pullingArms, realWorkers, experiment, workerset):
        rewards = np.zeros(para.nArms[workerset])

        for arm in range(para.nArms[workerset]):
            if pullingArms[arm] == 1 and realWorkers[arm] > 0:
                results = np.random.binomial(realWorkers[arm], np.array(para.rewardProbs[workerset], dtype=np.float64)[arm])
                rewards[arm] += results
      
        return rewards
    
    def reactWorkerNum(pullingArms, timeStamp, doneWorkers, experiment, workerset):
        
        workersPerTime = np.zeros(para.nArms[workerset])

        batchTime = (timeStamp - np.array(para.batchCreateTime[workerset], dtype=np.int32)[0]).copy()
        batchTime = np.where(batchTime <= 0, 0.5, batchTime)
        
        
        workersPerTime = (-1 * np.exp(-1 * np.array(para.theta0[workerset], dtype=np.float64) * batchTime + np.array(para.theta1[workerset], dtype=np.float64)) * np.array(para.theta2[workerset], dtype=np.float64) + np.array(para.theta3[workerset], dtype=np.float64)) * pullingArms
    

        workersPerTime = np.where(batchTime < 1, 0, workersPerTime)
        workersPerTime = workersPerTime.astype(np.int64)

        #print(pullingArms, batchTime, workersPerTime, 'workers')

        return workersPerTime
    



def feeCount(Workers, experiment, workerset):
    fee = Workers * np.array(para.feePara[workerset], dtype=np.int32)

    fee = np.maximum(fee, np.array(para.minimumFee[workerset], dtype=np.int32))

    fee[Workers == 0] = 0
    
    return np.sum(fee)


def feeCountSeperate(Workers, experiment, workerset):
    fee = Workers * np.array(para.feePara[workerset], dtype=np.int32)

    fee = np.maximum(fee, np.array(para.minimumFee[workerset], dtype=np.int32))

    fee[Workers == 0] = 0
    
    return fee


#print(env.reactWorkerNum(np.array([1, 1, 1]), np.array([4, 4, 4]), np.array([1, 1, 1])))