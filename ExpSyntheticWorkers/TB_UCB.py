#time-batch-multi-objective-ucb
import numpy as np
import time
from scipy.optimize import linprog
#from DesignBatch import designBatch
from Parameters import para

#from NonBatchedKnapsackwithGreedyDensity import rewardsRnb, tRnb, budgetRnb, pulledRnb

class TB_UCBalgorithm(object):
    def __init__(self, experiment, workerset):
        self.atomPullingCounts = np.zeros(para.nArms[workerset])
        self.atomReward = np.zeros(para.nArms[workerset])
        self.estimatedRewards = np.zeros(para.nArms[workerset])
        self.estimatedTraffic = np.zeros(para.nArms[workerset])
        self.delta = para.delta[experiment]
        self.probs = np.zeros(para.nArms[workerset])


    def get_arms(self, remainTime, remainBudget, estimatedTraffic, estimatedFee, estimatedDensityTraffic, estimatedDensityFee):
        
        self.estimatedRewards = self.atomReward / self.atomPullingCounts + np.sqrt((self.delta * np.log(np.sum(self.atomPullingCounts))) / self.atomPullingCounts)
        #print('check bandit:', self.estimatedRewards, self.atomPullingCounts)

        computedReward, pullArms = self.optimization(remainTime, remainBudget, estimatedTraffic, estimatedFee)
        #computedDensityReward, pullDensityArms = self.optimization(remainTime, remainBudget, estimatedDensityTraffic, estimatedDensityFee)

        cuttingMethod = 0

        #print(computedReward, computedDensityReward, 'compare')

        '''if computedDensityReward > computedReward:
            pullArms = pullDensityArms
            cuttingMethod = 1'''


        return pullArms, cuttingMethod
    
    def optimization(self, remainTime, remainBudget, estimatedTraffic, estimatedFee):
        c = -self.estimatedRewards * estimatedTraffic
        
        # minimize -c * x = -[traffic1 * reward1 * x1, traffic2 * reward2 * x2, ... , traffick * rewardk * xk]


        A = [estimatedFee]
        b = [remainBudget]
        #subject to Ax <= b which means fee1 * traffic1 * x1, fee2 * traffic2 * x2, ... , feek * traffick * xk


        bounds = (0, remainTime)
        # Bounds of x1, x2, ..., xk
    
        solution = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        eks = solution.x.copy()
        
        maxx = np.max(eks)
        normalizedSolution = solution.x / maxx
        
        return solution.fun, normalizedSolution


    def sample(self, pulledCounts, rewards):
        self.atomPullingCounts = pulledCounts
        self.atomReward = rewards
    
        

class oracle(object):
    def __init__(self):
        pass
    def get_arms(self, remainTime, remainBudget, estimatedTraffic, estimatedFee, estimatedDensityTraffic, estimatedDensityFee):
        return np.array([1000, 1000, 1000]), 0
    def sample(self, pulledCounts, rewards):
        pass






