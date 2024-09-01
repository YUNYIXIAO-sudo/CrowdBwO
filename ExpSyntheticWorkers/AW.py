import numpy as np
import time
from scipy.optimize import linprog
from Parameters import para
from environment import env, feeCount, feeCountSeperate
from NewTrafficFunction import estimateTraffic
from matplotlib import pyplot as plt
import pandas as pd

np.random.seed(32)
class AW(object):
    def __init__(self, experiment, workerset):
        self.pullArms = np.ones(para.nArms[workerset])
        self.exp = experiment
        self.workerset = workerset


    def get_arms(self):
        
        return self.pullArms


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

        for n in range(N): #number of simulations
            #Initializations
            agent = Agent(experiment, workerset)
            pullingIsFeasible = 1

            self.pullingArms = np.zeros(para.nArms[workerset])
            self.rewards = np.zeros(para.nArms[workerset])
            self.pulledCounts = np.zeros(para.nArms[workerset])
            self.pulledTime = np.zeros(para.nArms[workerset])


            self.t = 0

            self.usedBudget = 0
            
            pullingArms = agent.get_arms()
            
            self.pullingArms = pullingArms.copy()

            while pullingIsFeasible == 1:
        
                realWorkers = env.reactWorkerNum(pullingArms, self.t, np.sum(self.pulledCounts, axis=0), experiment, workerset)
                
                
                self.pulledCounts = realWorkers

                self.rewards = env.reactRewards(pullingArms, realWorkers, experiment, workerset)


                fee = feeCount(self.pulledCounts, experiment, workerset)
                self.usedBudget = fee

                        
                self.t += 1
                self.pulledTime += self.pullingArms
                
            

                if self.t >= T or self.usedBudget >= B:
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
    f = open('./expAVEResults/AW-S1averageResultsWS' + str(i) + '.csv', mode='w')
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
        selectedArmsR, rewardsR, pulledR, budgetR, tR = si.sim(AW)
        
        results = np.hstack((np.array(selectedArmsR), rewardsR, pulledR, np.array(budgetR), np.array(tR)))
        df = pd.DataFrame(results)
        #, columns=['Selected Project', 'Cumulative Rewards', 'Number of Assignments', 'Remaining Budget', 'Spent Time'])
        
        df.to_csv('./expResults/AW-S1resultsW' + str(workerset) + 'E' + str(experiment) + '.csv', mode='w')


        averageResults = np.hstack((np.mean(np.array(selectedArmsR), axis=0), np.mean(rewardsR, axis=0), np.mean(pulledR, axis=0), np.mean(np.array(budgetR), axis=0), np.mean(np.array(tR), axis=0)))
        df = pd.DataFrame([averageResults])
        
        df.to_csv('./expAVEResults/AW-S1averageResultsWS' + str(i) + '.csv', mode='a', index=False, header=False)




       