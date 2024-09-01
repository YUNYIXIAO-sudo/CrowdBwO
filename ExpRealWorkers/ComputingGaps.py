import numpy as np
from NewTrafficFuctions import estimateTraffic
from countTime import timeCountS1, timeCountS2
from AMTexperimentA.parameter import para
from matplotlib import pyplot as plt

target = [50, 30, 40]

def computeGap(setting, estimatedAccuracy1, estimatedAccuracy2, timeTrace, reward, usedBudget, means, stds, i):
    if setting == 1:
        estimatedTrafficPara, estimatedFeePara, mtime = estimateTraffic.getTraffic(para.timeLimit, means, stds)
       
    elif setting == 2:
        estimatedTrafficPara, estimatedFeePara, mtime = estimateTraffic.getTraffic(1000, means, stds)
        
    estimatedTrafficPara1 = estimatedTrafficPara[0]
    estimatedTrafficPara2 = estimatedTrafficPara[1]
    estimatedFeePara1 = estimatedFeePara[0]
    estimatedFeePara2 = estimatedFeePara[1]
    estimatedFeePara3 = estimatedFeePara1 + estimatedFeePara2

    estimatedRewardperTime1 = estimatedAccuracy1 * estimatedTrafficPara1
    estimatedRewardperTime2 = estimatedAccuracy2 * estimatedTrafficPara2
    estimatedRewardperTime3 = estimatedAccuracy1 * estimatedTrafficPara1 + estimatedAccuracy2 * estimatedTrafficPara2

    if setting == 1:
        t1, t2, t3 = timeCountS1(estimatedFeePara1, estimatedFeePara2)   
        maxIndex = np.argmax([estimatedRewardperTime1 * t1, estimatedRewardperTime2 * t2, estimatedRewardperTime3 * t3]) + 1
        print('OPTIMAL Worker Set(s) :', maxIndex, t1*estimatedAccuracy1*estimatedTrafficPara1, t2*estimatedAccuracy2*estimatedTrafficPara2, t3*(estimatedAccuracy1*estimatedTrafficPara1+estimatedAccuracy2*estimatedTrafficPara2))

        
    elif setting == 2:
        #reward[-1] = target[i]
        t1, t2, t3 = timeCountS2(estimatedAccuracy1, estimatedAccuracy2, estimatedTrafficPara1, estimatedTrafficPara2, target[i])
        cost1 = para.omega*estimatedFeePara1*t1 + (1 - para.omega)*t1
        cost2 = para.omega*estimatedFeePara2*t2 + (1 - para.omega)*t2
        cost3 = para.omega*estimatedFeePara3*t3 + (1 - para.omega)*t3
        maxIndex = np.argmin([cost1, cost2, cost3]) + 1


        print('OPTIMAL Worker Set(s) :', maxIndex, cost1, cost2, cost3, t1*estimatedAccuracy1*estimatedTrafficPara1, t2*estimatedAccuracy2*estimatedTrafficPara2, t3*(estimatedAccuracy1*estimatedTrafficPara1+estimatedAccuracy2*estimatedTrafficPara2))

    

    estimatedPullperTime1 = estimatedTrafficPara1
    estimatedPullperTime2 = estimatedTrafficPara2
    estimatedPullperTime3 = (estimatedTrafficPara1 + estimatedTrafficPara2)

    estimatedFeeperTime1 = estimatedFeePara1
    estimatedFeeperTime2 = estimatedFeePara2
    estimatedFeeperTime3 = (estimatedFeePara1 + estimatedFeePara2)


    optEstimatedRewardperTime = locals()['estimatedRewardperTime' + str(maxIndex)]
    optEstimatedPullperTime = locals()['estimatedPullperTime' + str(maxIndex)]
    optEstimatedFeeperTime = locals()['estimatedFeeperTime' + str(maxIndex)]
    optTime = locals()['t' + str(maxIndex)]

    

    if setting == 1:
        optRewardTrace = timeTrace * optEstimatedRewardperTime
        optFeeTrace = timeTrace * optEstimatedFeeperTime

        if timeTrace[-1] > optTime:
            print('more', optTime)
            
            optRewardTrace = np.where(timeTrace > optTime, optTime * optEstimatedRewardperTime, optRewardTrace)
            optFeeTrace = np.where(timeTrace > optTime, optTime * optEstimatedFeeperTime, optFeeTrace)


        elif timeTrace[-1] < optTime:
            print('less')

            optRewardTrace = np.append(optRewardTrace, np.arange(int(timeTrace[-1])+1, optTime+1, 1) * optEstimatedRewardperTime)
            reward = np.append(reward, np.ones(optTime - int(timeTrace[-1])) * reward[-1])
            
            optFeeTrace = np.append(optFeeTrace, np.arange(int(timeTrace[-1])+1, optTime+1, 1) * optEstimatedFeeperTime)
            usedBudget = np.append(usedBudget, np.ones(optTime - int(timeTrace[-1])) * usedBudget[-1])

            timeTrace = np.append(timeTrace, np.arange(int(timeTrace[-1]) + 1, optTime + 1, 1))

        elif timeTrace[-1] == optTime:
            print('equal')

        optRewardTrace_nonzero = np.where(reward==0, 0.001, optRewardTrace)
        
        gap = (reward / optRewardTrace_nonzero).round(2) * 100
        
        return optRewardTrace.astype(np.int64), reward.astype(np.int64), optFeeTrace, usedBudget, gap, timeTrace


    elif setting == 2:
        optTimeTrace = reward / optEstimatedRewardperTime
        optFeeTrace = optTimeTrace * optEstimatedFeeperTime
       

        if reward[-1] > target[i]:
            print('more')
            
            optTimeTrace = np.where(reward > target[i], optTime, optTimeTrace)
            optFeeTrace = np.where(reward > target[i], optTime * optEstimatedFeeperTime, optFeeTrace)
            

        elif reward[-1] < target[i]:
            print('less')
           
            optTimeTrace = np.append(optTimeTrace, np.arange(reward[-1]+1, target[i] + 1, 1) / optEstimatedRewardperTime)
            timeTrace = np.append(timeTrace, np.ones(int(target[i] - reward[-1])) * timeTrace[-1])
            
            optFeeTrace = np.append(optFeeTrace, np.arange(reward[-1]+1, target[i] + 1, 1) / optEstimatedRewardperTime * optEstimatedFeeperTime)
            usedBudget = np.append(usedBudget, np.ones(int(target[i] - reward[-1])) * usedBudget[-1])

            reward = np.append(reward, np.arange(reward[-1] + 1, target[i] + 1, 1))

        elif reward[-1] == target[i]:
            print('equal')

        
        #we don't need the data where real data's reward is 0
        timeTrace = np.delete(timeTrace, np.where(reward==0)[0])
        usedBudget = np.delete(usedBudget, np.where(reward==0)[0])
        optTimeTrace_nonzero = np.delete(optTimeTrace, np.where(reward==0)[0])
        optFeeTrace_nonzero = np.delete(optFeeTrace, np.where(reward==0)[0])
        reward_nonzero = np.delete(reward, np.where(reward==0)[0])



        gapTime = (timeTrace / optTimeTrace_nonzero).round(2) * 100
        gapFee = (usedBudget / optFeeTrace_nonzero).round(2) * 100
        gapCombi = ((para.omega * usedBudget + (1 - para.omega) * timeTrace) / (para.omega * optFeeTrace_nonzero + (1 - para.omega) * optTimeTrace_nonzero)).round(2) * 100
        #print('TEST ZERO', timeTrace, optTimeTrace, optTimeTrace_nonzero, gapCombi)

        return optTimeTrace_nonzero.astype(np.int64), timeTrace, optFeeTrace_nonzero, usedBudget, gapTime, gapFee, gapCombi, reward_nonzero.astype(np.int64)

    

    