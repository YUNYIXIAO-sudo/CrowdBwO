import numpy as np
from matplotlib import pyplot as plt
from Parameters import para
from environment import env, feeCount
import pandas as pd

np.random.seed(32)

class BwKGD(object):
    def __init__(self, experiment, workerset):
        self.atomPullingCounts = np.zeros(para.nArms[workerset])
        self.atomPullingTime = np.zeros(para.nArms[workerset])
        self.atomReward = np.zeros(para.nArms[workerset])
        self.estimatedRewards = np.zeros(para.nArms[workerset])
        self.delta = para.delta[experiment]
        self.probs = np.zeros(2**para.nArms[workerset])

    def estimateRewards(self):
        self.estimatedRewards = self.atomReward / self.atomPullingCounts + np.sqrt((0.3 * np.log(np.sum(self.atomPullingCounts))) / self.atomPullingCounts)

    def computeProbs(self): 
        self.estimateRewards()
        pullingArms = np.zeros(para.nArms[workerset])

        for a in range(2**para.nArms[workerset]):
            for b in range(para.nArms[workerset]):
                pullingArms[b] =  ((a >> b) & 1)

            if np.all(pullingArms == 0):
                self.probs[a] = 0
            else:
                self.probs[a] = np.dot(self.estimatedRewards, pullingArms) / feeCount(pullingArms, experiment, workerset)
            #computing the desity of each project



    def get_arms(self):
        self.computeProbs()
        
        pullingAction = np.argmax(self.probs)
        pullingArms = [((pullingAction >> i) & 1) for i in range(para.nArms[workerset])]
        
        return pullingArms
    
    def sample(self, pulledArms, rewards):
        self.atomPullingCounts += pulledArms
        self.atomReward += rewards




class simulation(object):
    def __init__(self):
        self.pulledTimeR = [para.nArms[workerset] * [0]]
        self.rewardsR = np.zeros(para.nArms[workerset])
        self.pulledCountsR = np.zeros(para.nArms[workerset])
        self.BudgetR = [[0]]
        self.tR = [[0]]


    def sim(self, Agent):
        N=para.n_iter[experiment]
        B=para.budget[experiment]
        T=para.timeLimit[experiment]
        requiredResults = para.answersRequired[experiment]

        for n in range(N): #number of simulations
            agent = Agent(experiment, workerset)
            pullingIsFeasible = 1
            self.pullingArms = np.zeros(para.nArms[workerset])
            self.rewards = np.zeros(para.nArms[workerset])
            self.pulledCounts = np.zeros(para.nArms[workerset])
            self.pulledTime = np.zeros(para.nArms[workerset])

            self.t = 0
            self.m = 0
            self.M = 0
            self.usedBudget = 0
            


            while pullingIsFeasible == 1:
                self.m += 1

                if self.m <= para.nArms[workerset]:
                    pullingArms = np.ones(para.nArms[workerset])

                elif self.m > para.nArms[workerset]:
                    pullingArms = agent.get_arms()
                
                self.pullingArms = np.array(pullingArms).copy()

                batchTime = 0
                getWorker = 0

                while getWorker == 0:
                    batchTime += 1
                    
                    realWorkers = env.reactWorkerNum(pullingArms, batchTime, self.pulledCounts, experiment, workerset)
                    
                    if np.all(realWorkers[np.array(pullingArms) > 0] >= 1):
                        getWorker = 1

                
                self.t += batchTime
                self.pulledTime += self.pullingArms * batchTime

                fee = feeCount(pullingArms, experiment, workerset)
                self.usedBudget += fee
                self.pulledCounts += pullingArms

                rewards = env.reactRewards(pullingArms, np.ones(para.nArms[workerset]), experiment, workerset)
                
                self.rewards += rewards

                agent.sample(pullingArms, rewards)
                    
                
                if np.sum(self.rewards) >= requiredResults or self.t >= para.S2timeLimit[experiment]:
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
    f = open('./expAVEResults/BwK-GDaverageResultsWS' + str(i) + '.csv', mode='w')
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
        selectedArmsR, rewardsR, pulledR, budgetR, tR = si.sim(BwKGD)

        results = np.hstack((np.array(selectedArmsR), rewardsR, pulledR, np.array(budgetR), np.array(tR)))
        df = pd.DataFrame(results)
        #, columns=['Selected Project', 'Cumulative Rewards', 'Number of Assignments', 'Remaining Budget', 'Spent Time'])
        
        df.to_csv('./expResults/BwK-GDresultsW' + str(workerset) + 'E' + str(experiment) + '.csv', mode='w')

        averageResults = np.hstack((np.mean(np.array(selectedArmsR), axis=0), np.mean(rewardsR, axis=0), np.mean(pulledR, axis=0), np.mean(np.array(budgetR), axis=0), np.mean(np.array(tR), axis=0)))
        df = pd.DataFrame([averageResults])
        
        df.to_csv('./expAVEResults/BwK-GDaverageResultsWS' + str(i) + '.csv', mode='a', index=False, header=False)


