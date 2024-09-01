#time-batch-multi-objective-ucb
import numpy as np
import time
from scipy.optimize import linprog
from parameter import para

#from NonBatchedKnapsackwithGreedyDensity import rewardsRnb, tRnb, budgetRnb, pulledRnb

class TB_UCBalgorithm(object):
    def __init__(self):
        self.atomPullingCounts = np.zeros(para.nArms)
        self.atomReward = np.zeros(para.nArms)
        self.estimatedRewards = np.zeros(para.nArms)
        self.estimatedTraffic = np.zeros(para.nArms)
        self.delta = 0.5
        self.probs = np.zeros(para.nArms)


    def get_arms(self, remainTime, remainBudget, estimatedTraffic, estimatedFee):
        
        self.atomPullingCounts = np.where(self.atomPullingCounts == 0, 0.1, self.atomPullingCounts)
        
        self.estimatedRewards = self.atomReward / self.atomPullingCounts + np.sqrt((self.delta * np.log(np.sum(self.atomPullingCounts))) / self.atomPullingCounts)


        computedReward, pullArms = self.optimization(remainTime, remainBudget, estimatedTraffic, estimatedFee)
        #computedDensityReward, pullDensityArms = self.optimization(remainTime, remainBudget, estimatedDensityTraffic, estimatedDensityFee)


        #print(computedReward, computedDensityReward, 'compare')

        '''if computedDensityReward > computedReward:
            pullArms = pullDensityArms
            cuttingMethod = 1'''

        return pullArms


    def optimization(self, remainTime, remainBudget, estimatedTraffic, estimatedFee):
        c = -self.estimatedRewards * estimatedTraffic
        # minimize -c * x = -[traffic1 * reward1 * x1, traffic2 * reward2 * x2, ... , traffick * rewardk * xk]


        A = [estimatedFee]
        b = [remainBudget]
        #subject to Ax <= b which means fee1 * traffic1 * x1, fee2 * traffic2 * x2, ... , feek * traffick * xk


        bounds = (0, remainTime)
        # Bounds of x1, x2, ..., xk
    
        solution = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        
        normalizedSolution = solution.x / np.max(solution.x)
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






