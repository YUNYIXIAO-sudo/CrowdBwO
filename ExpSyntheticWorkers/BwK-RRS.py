import numpy as np
from matplotlib import pyplot as plt
from Parameters import para
from environment import env, feeCountSeperate
import pandas as pd
from scipy.optimize import linprog

np.random.seed(32)

#alpha and epsilon are parameters only for semibwk
alpha = 0.2


class SemiBwK(object):
    def __init__(self, TotalRounds, Budget, experiment, workerset, epsilon=0.99):
        self.atomPullingCounts = np.zeros(para.nArms[workerset])
        self.atomReward = np.zeros(para.nArms[workerset])
        self.atomCost = np.zeros(para.nArms[workerset])
        self.atomTimeCost = np.zeros(para.nArms[workerset])
        self.estimatedRewards = np.zeros(para.nArms[workerset])
        self.estimatedCosts = np.zeros(para.nArms[workerset])
        self.estimatedTimeCosts = np.zeros(para.nArms[workerset])
        self.delta = para.delta[experiment]
        self.probs = np.zeros(2**para.nArms[workerset])
        self.roundBudget = (Budget * epsilon) / TotalRounds

        

    def estimateRewards(self, alpha=alpha):
        beforeProj = self.atomReward / self.atomPullingCounts + np.sqrt(alpha * (self.atomReward / self.atomPullingCounts) / self.atomPullingCounts) + (self.atomReward / self.atomPullingCounts) / self.atomPullingCounts
              
        self.estimatedRewards = np.where(beforeProj >= 1 - beforeProj, 1, 0)
        

    def estimateCosts(self, alpha=alpha):
        beforeProj = self.atomCost / self.atomPullingCounts - (np.sqrt(alpha * self.atomCost / self.atomPullingCounts / self.atomPullingCounts) + self.atomCost / self.atomPullingCounts / self.atomPullingCounts)
        
        self.estimatedCosts = np.where(beforeProj >= 1 - beforeProj, 1, 0)

    def estimateTimeCosts(self, alpha=alpha):
        beforeProj = self.atomTimeCost / self.atomPullingCounts - (np.sqrt(alpha * self.atomTimeCost / self.atomPullingCounts / self.atomPullingCounts) + self.atomTimeCost / self.atomPullingCounts / self.atomPullingCounts)

        self.estimatedTimeCosts = np.where(beforeProj >= 1 - beforeProj, 1, 0)
        


    def get_arms(self):
        self.estimateRewards()
        self.estimateCosts()
        self.estimateTimeCosts()
        

        c = -self.estimatedRewards
        # minimize -c * x = -[traffic1 * reward1 * x1, traffic2 * reward2 * x2, ... , traffick * rewardk * xk]


        A = [self.estimatedCosts, self.estimatedTimeCosts]
        b = [self.roundBudget, self.roundBudget]
        #subject to Ax <= b which means fee1 * traffic1 * x1, fee2 * traffic2 * x2, ... , feek * traffick * xk


        bounds = (0, 1)
        # Bounds of x1, x2, ..., xk
    
        solution = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
       
        return solution.x
    
    def sample(self, pulledArms, rewards, cost, timeCost):
        self.atomPullingCounts = pulledArms
        self.atomReward = rewards
        self.atomCost = cost
        self.atomTimeCost = timeCost
        


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
        TotalRounds = para.totalRoundsBwKS1[workerset][experiment]
        budget = min(B, T)
        rateofBudget = B/T
     
        

        for n in range(N): #number of simulations
            agent = Agent(TotalRounds, budget, experiment, workerset)
            pullingIsFeasible = 1
            self.pullingArms = np.zeros(para.nArms[workerset])
            self.rewards = np.zeros(para.nArms[workerset])
            self.pulledCounts = np.zeros(para.nArms[workerset])
            self.pulledTime = np.zeros(para.nArms[workerset])

            self.t = 0
            self.m = 0
            self.M = 0
            self.usedBudget = 0
            self.Costs = np.zeros(para.nArms[workerset])
            self.timeCosts = np.zeros(para.nArms[workerset])



            roundIndex = 0

            while pullingIsFeasible == 1:
                self.m += 1

                if self.m <= para.nArms[workerset]:
                    pullingArmsThisTime = np.ones(para.nArms[workerset])
                    pullingArms = np.ones(para.nArms[workerset])
                    

                elif self.m > para.nArms[workerset]:
                    pullingArms = agent.get_arms()
                    pullingArmsThisTime = np.ones(para.nArms[workerset])
                    pullingArmsThisTime = [np.random.binomial(1, p) for p in pullingArms]
                   

                self.pullingArms = np.array(pullingArmsThisTime).copy()

                batchTime = 0
                getWorker = 0
                indextoPull = np.ones(para.nArms[workerset])
                indextoPull = indextoPull * pullingArmsThisTime

                while getWorker == 0:
                    if np.all(indextoPull == 0) == 0:
                        realWorkers = env.reactWorkerNum(pullingArmsThisTime, batchTime, self.pulledCounts, experiment, workerset)

                        self.timeCosts += indextoPull
                        
                        indextoPull[realWorkers > 1] = 0

                        batchTime += 1

                    else: 
                        getWorker = 1
            
                self.t += batchTime
                self.pulledTime += self.pullingArms * batchTime
                roundIndex += 1

                fee = feeCountSeperate(np.ones(para.nArms[workerset]), experiment, workerset) 
                
                self.usedBudget += np.sum(fee)
                self.Costs += fee / rateofBudget

                self.pulledCounts += pullingArmsThisTime

                rewards = env.reactRewards(pullingArmsThisTime, np.ones(para.nArms[workerset]), experiment, workerset)
               
                self.rewards += rewards


                agent.sample(self.pulledCounts, self.rewards, self.Costs, self.timeCosts)


                if roundIndex >= TotalRounds:
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
    f = open('./expAVEResults/BwK-RRSaverageResultsWS' + str(i) + '.csv', mode='w')
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
        selectedArmsR, rewardsR, pulledR, budgetR, tR = si.sim(SemiBwK)


        #print(np.array(selectedArmsRnb), rewardsRnb, pulledRnb, np.array(budgetRnb), np.array(tRnb))
        results = np.hstack((np.array(selectedArmsR), rewardsR, pulledR, np.array(budgetR), np.array(tR)))
        df = pd.DataFrame(results)
        #, columns=['Selected Project', 'Cumulative Rewards', 'Number of Assignments', 'Remaining Budget', 'Spent Time'])
        
        df.to_csv('./expResults/BwK-RRSresultsW' + str(workerset) + 'E' + str(experiment) + '.csv', mode='w')

        averageResults = np.hstack((np.mean(np.array(selectedArmsR), axis=0), np.mean(rewardsR, axis=0), np.mean(pulledR, axis=0), np.mean(np.array(budgetR), axis=0), np.mean(np.array(tR), axis=0)))
        print('final results:', averageResults.astype(int))
        df = pd.DataFrame([averageResults])
        
        df.to_csv('./expAVEResults/BwK-RRSaverageResultsWS' + str(i) + '.csv', mode='a', index=False, header=False)


