import numpy as np
import time
from scipy.optimize import linprog
from Parameters import para
from environment import env, feeCount, feeCountSeperate
from matplotlib import pyplot as plt
from scipy import optimize
import pandas as pd
from NewTrafficFunction import estimateTraffic

np.random.seed(32)
class OWOP(object):
    def __init__(self, experiment, workerset):
        self.pullArms = np.zeros(para.nArms[workerset])
        self.exp = experiment
        self.workerset = workerset
        

        theta0 = np.array(para.theta0[workerset], dtype=np.float64)
        theta1 = np.array(para.theta1[workerset], dtype=np.float64)
        theta2 = np.array(para.theta2[workerset], dtype=np.float64)
        theta3 = np.array(para.theta3[workerset], dtype=np.float64)

        self.thetas = np.vstack((theta0, theta1, theta2, theta3))
        
        
    def optimization_LP(self):
        T = para.timeLimit[self.exp]
        B = para.budget[self.exp]
        estimatedTraffic, estimatedFee, self.time = estimateTraffic.getTraffic(T, self.thetas, np.zeros(shape=(4, para.nArms[self.workerset])), self.exp, self.workerset)
        
        c = -np.array(para.rewardProbs[workerset], dtype=np.float64) * estimatedTraffic
        # minimize -c * x = -[traffic1 * reward1 * x1, traffic2 * reward2 * x2, ... , traffick * rewardk * xk]

        A = [estimatedFee]
        b = [B]
        #subject to Ax <= b which means fee1 * traffic1 * x1, fee2 * traffic2 * x2, ... , feek * traffick * xk


        bounds = (0, T)
        # Bounds of x1, x2, ..., xk
    
        solution = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        self.pullArms = solution.x
    
    
    def optimization_MO(self):
        requiredResults = para.answersRequired[self.exp]

        estimatedTraffic, estimatedFee, self.time = estimateTraffic.getTrafficNoTimeLimit(self.thetas, np.zeros(shape=(4, para.nArms[self.workerset])), self.exp, self.workerset)
        
        omega=para.omega[self.exp]

        objectiveFunc = lambda x: np.dot(omega * estimatedFee, x) + (1 - omega) * np.max(x)

        constraints = ({'type': 'ineq', 'fun': lambda x: np.dot(estimatedTraffic * np.array(para.rewardProbs[workerset], dtype=np.float64), x) - requiredResults})

        bounds = [(0, None) for _ in range(para.nArms[self.workerset])]

        firstGuess = np.ones(para.nArms[self.workerset])

        solution = optimize.minimize(objectiveFunc, firstGuess, method='SLSQP', bounds=bounds, constraints=constraints)

        self.pullArms = np.array(solution.x).astype(np.int64)
        
                    
        

    def get_arms(self):
        
        self.optimization_MO()
        return self.pullArms, self.time
    


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
            self.pulledTime = np.zeros(para.nArms[workerset])

            realWorkers = np.zeros(para.nArms[workerset])
            rewards = np.zeros(para.nArms[workerset])
            addWorkersforRecords = np.zeros(para.nArms[workerset])
            addRewardsforRecords = np.zeros(para.nArms[workerset])
            addFeeforRecords = 0


            self.t = 0
            self.usedBudget = 0

            pullingArms, mtime = agent.get_arms()
            

            while pullingIsFeasible == 1:
                normalizedprob = pullingArms / np.max(pullingArms)
                self.pullingArms = np.array([np.random.binomial(1, p) for p in normalizedprob])
              
                timeStamp = self.t % mtime 
                timeStamp[timeStamp == 0] = mtime[timeStamp == 0]
                timeStamp = timeStamp * self.pullingArms
                

                if np.any(timeStamp > np.array(para.batchCreateTime[workerset], dtype=np.int32)[0]) == 1:
                   

                    realWorkers = env.reactWorkerNum(self.pullingArms, timeStamp, np.sum(self.pulledCounts, axis=0), experiment, workerset)
                    self.pulledCounts[timeStamp == mtime] += realWorkers[timeStamp == mtime]
                    addWorkersforRecords = np.where(timeStamp == mtime, 0, realWorkers)
                    

                    rewards = env.reactRewards(self.pullingArms, realWorkers, experiment, workerset)
                    self.rewards[timeStamp == mtime] += rewards[timeStamp == mtime]
                    addRewardsforRecords = np.where(timeStamp == mtime, 0, rewards)
                        

                    fee = feeCountSeperate(realWorkers, experiment, workerset)
                    feetoCount = np.where(timeStamp != mtime, 0, fee)
                    self.usedBudget += np.sum(feetoCount)
                    addFeeforRecords = np.sum(np.where(timeStamp == mtime, 0, fee))

                        
                self.t += 1
                self.pulledTime += self.pullingArms
              
                print(self.pullingArms, np.sum(self.rewards), self.t, self.usedBudget)

                
                if np.sum(self.rewards) >= RequiredResult or self.t >= para.S2timeLimit[experiment]:
                    pullingIsFeasible = 0
                    self.recordsPrint(addWorkersforRecords, addRewardsforRecords, addFeeforRecords)

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
    f = open('./expAVEResults/OWOP-S2averageResultsWS' + str(i) + '.csv', mode='w')
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
        selectedArmsR, rewardsR, pulledR, budgetR, tR = si.sim(OWOP)
        
        results = np.hstack((np.array(selectedArmsR), rewardsR, pulledR, np.array(budgetR), np.array(tR)))
        df = pd.DataFrame(results)
        #, columns=['Selected Project', 'Cumulative Rewards', 'Number of Assignments', 'Remaining Budget', 'Spent Time'])
        
        df.to_csv('./expResults/OWOP-S2resultsW' + str(workerset) + 'E' + str(experiment) + '.csv', mode='w')

        averageResults = np.hstack((np.mean(np.array(selectedArmsR), axis=0), np.mean(rewardsR, axis=0), np.mean(pulledR, axis=0), np.mean(np.array(budgetR), axis=0), np.mean(np.array(tR), axis=0)))
        df = pd.DataFrame([averageResults])
        
        df.to_csv('./expAVEResults/OWOP-S2averageResultsWS' + str(i) + '.csv', mode='a', index=False, header=False)
