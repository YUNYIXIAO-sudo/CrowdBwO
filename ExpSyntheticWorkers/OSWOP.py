import numpy as np
import time
from Parameters import para
from environment import env, feeCount, feeCountSeperate
from matplotlib import pyplot as plt
import pandas as pd
from NewTrafficFunction import estimateTraffic

np.random.seed(32)
class OSWOP(object):
    def __init__(self, experiment, workerset):
        self.pullArms = np.zeros(para.nArms[workerset])
        self.exp = experiment
        self.workerset = workerset

        theta0 = np.array(para.theta0[workerset], dtype=np.float64)
        theta1 = np.array(para.theta1[workerset], dtype=np.float64)
        theta2 = np.array(para.theta2[workerset], dtype=np.float64)
        theta3 = np.array(para.theta3[workerset], dtype=np.float64)

        self.thetas = np.vstack((theta0, theta1, theta2, theta3))
    
    def S1(self):
        T = para.timeLimit[self.exp]
        B = para.budget[self.exp]
    
        AccumulatedWorkers = np.zeros(para.nArms[workerset])
        
        estimatedTraffic, estimatedFee, self.time = estimateTraffic.getTraffic(T, self.thetas, np.zeros(shape=(4, para.nArms[self.workerset])), self.exp, self.workerset)
        
        for t in range(T):
            
            workers = estimatedTraffic * t
            fees = estimatedFee * t
            AccumulatedWorkers[fees <= B] = workers[fees <= B]
        
        AccumulatedRewards = (AccumulatedWorkers * np.array(para.rewardProbs[workerset], dtype=np.float64)).copy()

        self.pullArms[np.argmax(AccumulatedRewards)] = 1
      

    def S2(self):
        RequiredResults = para.answersRequired[self.exp]

        AccumulatedRewards = np.zeros(para.nArms[workerset])
        usedTime = np.zeros(para.nArms[workerset])
        fees = np.zeros(para.nArms[workerset])

        t = 0
        estimatedTraffic, estimatedFee, self.time = estimateTraffic.getTrafficNoTimeLimit(self.thetas, np.zeros(shape=(4, para.nArms[workerset])), self.exp, self.workerset)

        while np.any(AccumulatedRewards < RequiredResults) == 1:
            t += 1
            workers = workers = estimatedTraffic * t
            AccumulatedRewards = (workers * np.array(para.rewardProbs[workerset], dtype=np.float64)).copy()
            fees[AccumulatedRewards <= RequiredResults] = (estimatedFee * t)[AccumulatedRewards <= RequiredResults]
            usedTime[AccumulatedRewards <= RequiredResults] = t

        self.pullArms[np.argmin(para.omega[self.exp] * usedTime + (1 - para.omega[self.exp]) * fees)] = 1

        
    def getCutTime(self):
        return self.time[self.pullArms == 1]


    def get_arms(self):
        self.S1()
        #self.S2()
        mtime = self.getCutTime()
        return self.pullArms, mtime
    


class simulation(object):
    def __init__(self):
        self.pulledTimeR = [para.nArms[workerset] * [0]]
        self.rewardsR = np.zeros(para.nArms[workerset])
        self.pulledCountsR = np.zeros(para.nArms[workerset])
        self.BudgetR = [[0]]
        self.tR = [[0]]


    def sim(self, Agent):
        N = para.n_iter[experiment]
        
        B = para.budget[experiment]
        T = para.timeLimit[experiment]
        RequiredResult = para.answersRequired[experiment]

        for n in range(N): #number of simulations
            #Initializations
            agent = Agent(experiment, workerset)
            pullingIsFeasible = 1

            self.pullingArms = np.zeros(para.nArms[workerset])
            self.rewards = np.zeros(para.nArms[workerset])
            self.pulledCounts = np.zeros(para.nArms[workerset])


            realWorkers = np.zeros(para.nArms[workerset])
            rewards = np.zeros(para.nArms[workerset])
            addWorkersforRecords = np.zeros(para.nArms[workerset])
            addRewardsforRecords = np.zeros(para.nArms[workerset])
            addFeeforRecords = 0
            self.pulledTime = np.zeros(para.nArms[workerset])


            self.t = 0
            self.usedBudget = 0

            pullingArms, mtime = agent.get_arms()
            self.pullingArms = pullingArms.copy()

            while pullingIsFeasible == 1:
                timeStamp = self.t % mtime 
                timeStamp[timeStamp == 0] = mtime[timeStamp == 0]
                timeStamp = timeStamp * pullingArms
                

                if np.any(timeStamp > np.array(para.batchCreateTime[workerset], dtype=np.int32)[0]) == 1:
                   

                    realWorkers = env.reactWorkerNum(pullingArms, timeStamp, np.sum(self.pulledCounts, axis=0), experiment, workerset)
                    self.pulledCounts[timeStamp == mtime] += realWorkers[timeStamp == mtime]
                    addWorkersforRecords = np.where(timeStamp == mtime, 0, realWorkers)
                    

                    rewards = env.reactRewards(pullingArms, realWorkers, experiment, workerset)
                    self.rewards[timeStamp == mtime] += rewards[timeStamp == mtime]
                    addRewardsforRecords = np.where(timeStamp == mtime, 0, rewards)
                        

                    fee = feeCountSeperate(realWorkers, experiment, workerset)
                    feetoCount = np.where(timeStamp != mtime, 0, fee)
                    self.usedBudget += np.sum(feetoCount)
                    addFeeforRecords = np.sum(np.where(timeStamp == mtime, 0, fee))

                        
                self.t += 1
                self.pulledTime += self.pullingArms

                
                if self.t >= T or self.usedBudget + addFeeforRecords >= B:
                    pullingIsFeasible = 0
                    self.recordsPrint(addWorkersforRecords, addRewardsforRecords, addFeeforRecords)

                   

                '''if np.sum(self.rewards) >= RequiredResult:
                    pullingIsFeasible = 0'''
                

        self.pulledTimeR = np.delete(self.pulledTimeR, 0, 0)
        self.rewardsR = np.delete(self.rewardsR, 0, 0)
        self.pulledCountsR = np.delete(self.pulledCountsR, 0, 0)
        self.BudgetR = np.delete(self.BudgetR, 0, 0)
        self.tR = np.delete(self.tR, 0, 0)

        return self.pulledTimeR, self.rewardsR, self.pulledCountsR, self.BudgetR, self.tR
            
    def recordsPrint(self, pulled, rewards, fee):
        self.pulledTimeR.append(self.pulledTime)
        self.rewardsR = np.vstack((self.rewardsR, self.rewards + rewards))
        self.pulledCountsR = np.vstack((self.pulledCountsR, self.pulledCounts + pulled))
        self.BudgetR.append([self.usedBudget + fee])
        self.tR.append([self.t])
        



for i in range(para.workerGroupsNumber):    
    workerset = i
    f = open('./expAVEResults/OSWOP-S1averageResultsWS' + str(i) + '.csv', mode='w')
    index = ''
    for a in range(para.nArms[i]*3+2):
        index += str(a)
        index += ','
    index = index[:-1]
    index += '\n'
    
    f.write(index)
    f.close()

    for j in range(para.expNumber):
        experiment = j

        si = simulation()
        selectedArmsR, rewardsR, pulledR, budgetR, tR = si.sim(OSWOP)
        
        results = np.hstack((np.array(selectedArmsR), rewardsR, pulledR, np.array(budgetR), np.array(tR)))
        df = pd.DataFrame(results)
        #, columns=['Selected Project', 'Cumulative Rewards', 'Number of Assignments', 'Remaining Budget', 'Spent Time'])
        
        df.to_csv('./expResults/OSWOP-S1resultsW' + str(workerset) + 'E' + str(experiment) + '.csv', mode='w')

        averageResults = np.hstack((np.mean(np.array(selectedArmsR), axis=0), np.mean(rewardsR, axis=0), np.mean(pulledR, axis=0), np.mean(np.array(budgetR), axis=0), np.mean(np.array(tR), axis=0)))
        df = pd.DataFrame([averageResults])
        
        df.to_csv('./expAVEResults/OSWOP-S1averageResultsWS' + str(i) + '.csv', mode='a', index=False, header=False)

