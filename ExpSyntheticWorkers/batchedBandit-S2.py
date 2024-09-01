#time-batch-multi-objective-ucb
import numpy as np
import time
from Parameters import para
from matplotlib import pyplot as plt
from environment import env
import pandas as pd
from Parameters import para
import math
from environment import env, feeCount

np.random.seed(32)
class batchedBanditalgorithm(object):
    def __init__(self, experiment, workerset):
        self.atomPullingCounts = np.zeros(para.nArms[workerset])
        self.atomReward = np.zeros(para.nArms[workerset])
        self.estimatedRewards = np.zeros(para.nArms[workerset])
        self.estimatedTraffic = np.zeros(para.nArms[workerset])
        self.delta = para.delta[experiment]
        self.probs = np.zeros(para.nArms[workerset])
        self.K = para.nArms[workerset]
        self.T = para.timeLimit[experiment]


    def get_arms(self):
        eta = 0.2
        #eta is a special parameter only for batched bandit algorithm

        self.estimatedRewards = self.atomReward / self.atomPullingCounts

        confidenceBound = np.sqrt((eta * np.log(self.T * self.K)) / np.max(self.atomPullingCounts))
       
        maxReward = np.max(self.estimatedRewards)

        pullArms = np.where(maxReward - self.estimatedRewards >= confidenceBound, 0, np.ones(self.estimatedRewards.shape))

        return pullArms 


    def sample(self, pulledCounts, rewards):
        self.atomPullingCounts = pulledCounts
        self.atomReward = rewards


def geometricGrids(batchSizeLast, beta):
    batchSize = math.floor(beta * batchSizeLast)

    return batchSize

def minimaxGrids(batchSizeLast, alpha):
    batchSize = math.floor(alpha * np.sqrt(batchSizeLast))

    return batchSize




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
        totalRounds = para.totalRoundsBBS2[workerset][experiment]

        for n in range(N): #number of simulations
            #Initializations
            agent = Agent(experiment, workerset)
            pullingIsFeasible = 1

            numberofArms = para.nArms[workerset]
            self.pullingArms = np.zeros(numberofArms)
            self.rewards = np.zeros(numberofArms)
            self.rewardsforEstimate = np.zeros(numberofArms)
            self.pulledCounts = np.zeros(numberofArms)
            self.pulledCountsforEstimate = np.zeros(numberofArms)
            self.pulledTime = np.zeros(para.nArms[workerset])
            

            self.t = 0
            self.timeIndex = 0
            self.m = 0
            self.tau = 0
            self.usedBudget = 0
        

            M = int(np.log2(np.log2(totalRounds)))
            alpha = int(totalRounds ** (1 / (2 - (2 ** (1 - M)))))
            
            print(M, 'm')

            #Initializations end

            #round loops
            while pullingIsFeasible == 1:
                #logging.info(f"Iteration {self.timeIndex}")

                if self.timeIndex == self.tau and self.m <= M:

                    if self.m == 0:
                        pullingArmsThisTime = np.ones(numberofArms)

                        #nextTau = geometricGrids(1, beta)
                        nextTau = minimaxGrids(1, alpha)
                        

                    elif self.m > 0:
                        
                        agent.sample(self.pulledCountsforEstimate, self.rewardsforEstimate)
                    
                        
                        #nextTau = geometricGrids(self.tau, beta)
                        nextTau = minimaxGrids(self.tau, alpha)
            

                        fee = feeCount(pullNumThisTime, experiment, workerset)
                        
                        
                        self.usedBudget += fee

                        
                        pullingArmsThisTime = agent.get_arms()
                     


                    grid = nextTau - self.tau
                        
                    self.pullingArms = pullingArmsThisTime.copy()

                    numPullingArm = np.sum(pullingArmsThisTime)
                    #number of arms to be pull in this batch

                    left = grid % numPullingArm
                    pullperArm = grid // numPullingArm
              
                    #times that cannot be divided perfectly into number of arms

                
                    self.tau = nextTau

                    batchTime = 0
                    workersThisTime = np.zeros(para.nArms[workerset])



                       
                if self.timeIndex == (self.tau - left) and left > 0:
                    
    
                    lastpull = pullingArmsThisTime * pullperArm
                    indexofPullingArms = np.where(self.pullingArms == 1)[0][0]
                    #only pull the first arm for the rest of time 
                    pullArmforLeft = np.zeros(para.nArms[workerset])
                    pullArmforLeft[indexofPullingArms] = 1
                    

                    while workersThisTime[indexofPullingArms] < left + (pullperArm):
                        batchTime += 1
                        self.t += 1
                        self.pulledTime += pullArmforLeft

                        realWorkers = env.reactWorkerNum(pullArmforLeft, batchTime, 0, experiment, workerset)
        
                        workersThisTime = realWorkers
                        #print('test', left + (pullperArm), workersThisTime, realWorkers, indexofPullingArms, pullArmforLeft, batchTime)
                       

                    self.pulledCounts[indexofPullingArms] += left

                    leftPull = np.array(pullArmforLeft) * left
                  
                    pullNumThisTime = lastpull + leftPull
          
                    self.rewards += env.reactRewards(pullArmforLeft, leftPull, experiment, workerset)
                    
                    self.timeIndex += left

                    self.m += 1


                else:
                    
                    batchTime += 1
                    
                    realWorkers = env.reactWorkerNum(pullingArmsThisTime, batchTime, 0, experiment, workerset)
                    
                    workersThisTime = realWorkers


                    if np.min(workersThisTime[pullingArmsThisTime > 0]) >= pullperArm:
                        pullNumThisTime = pullperArm * pullingArmsThisTime
                       
                        workersThisTime = pullNumThisTime
                        self.pulledCounts += pullNumThisTime
                        self.pulledCountsforEstimate += pullNumThisTime

                        reward = env.reactRewards(pullingArmsThisTime, pullNumThisTime, experiment, workerset)

                        self.rewards += reward
                        self.rewardsforEstimate += reward
                        self.timeIndex += (grid - left)
                    

                    self.t += np.sum(np.where(workersThisTime >= pullperArm, 0, pullingArmsThisTime))
                    self.pulledTime += np.where(workersThisTime >= pullperArm, 0, pullingArmsThisTime)
                
                
                
                if np.sum(self.rewards) >= para.answersRequired[experiment] or self.m >= M:
 
                    fee = feeCount(pullNumThisTime, experiment, workerset)
                    self.usedBudget += fee
                    pullingIsFeasible = 0
                    self.recordsPrint()


        self.pulledTimeR = np.delete(self.pulledTimeR, 0, 0)
        self.rewardsR = np.delete(self.rewardsR, 0, 0)
        self.pulledCountsR = np.delete(self.pulledCountsR, 0, 0)
        self.BudgetR = np.delete(self.BudgetR, 0, 0)
        self.tR = np.delete(self.tR, 0, 0)


        return self.pulledTimeR, self.rewardsR, self.pulledCountsR, self.BudgetR, self.tR
            
    def recordsPrint(self):
        self.pulledTimeR.append(self.pulledTime)
        self.rewardsR = np.vstack((self.rewardsR, self.rewards))
        self.pulledCountsR = np.vstack((self.pulledCountsR, self.pulledCounts))
        self.BudgetR.append([self.usedBudget])
        self.tR.append([self.t])
        



for i in range(para.workerGroupsNumber):    
    workerset = i
    f = open('./expAVEResults/BB-S2averageResultsWS' + str(i) + '.csv', mode='w')
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
        selectedArmsR, rewardsR, pulledR, budgetR, tR = si.sim(batchedBanditalgorithm)

        
        
        results = np.hstack((np.array(selectedArmsR), rewardsR, pulledR, np.array(budgetR), np.array(tR)))
        df = pd.DataFrame(results)
        #, columns=['Selected Project', 'Cumulative Rewards', 'Number of Assignments', 'Remaining Budget', 'Spent Time'])
        
        df.to_csv('./expResults/BB-S2resultsW' + str(workerset) + 'E' + str(experiment) + '.csv', mode='w')

        averageResults = np.hstack((np.mean(np.array(selectedArmsR), axis=0), np.mean(rewardsR, axis=0), np.mean(pulledR, axis=0), np.mean(np.array(budgetR), axis=0), np.mean(np.array(tR), axis=0)))
        print('final results:', averageResults)
        df = pd.DataFrame([averageResults])
        
        df.to_csv('./expAVEResults/BB-S2averageResultsWS' + str(i) + '.csv', mode='a', index=False, header=False)


