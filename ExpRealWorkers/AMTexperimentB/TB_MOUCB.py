#time-batch-multi-objective-ucb
import numpy as np
import time
from scipy import optimize
from matplotlib import pyplot as plt
#from DesignBatch import designBatch
from parameter import para


class TB_MOUCBalgorithm(object):
    def __init__(self):
        self.atomPullingCounts = np.zeros(para.nArms)
        self.atomReward = np.zeros(para.nArms)
        self.estimatedRewards = np.zeros(para.nArms)
        self.estimatedTraffic = np.zeros(para.nArms)
        self.delta = 0.5
        self.probs = np.zeros(para.nArms)
        


    def get_arms(self, remainAnswers, estimatedTraffic, estimatedFee):
        
        self.atomPullingCounts = np.where(self.atomPullingCounts == 0, 0.1, self.atomPullingCounts)
        
        self.estimatedRewards = self.atomReward / self.atomPullingCounts + np.sqrt((self.delta * np.log(np.sum(self.atomPullingCounts))) / self.atomPullingCounts)


        pullArms = self.MultiObjectiveOptimization(remainAnswers, estimatedTraffic, estimatedFee)
        #computedDensityReward, pullDensityArms = self.optimization(remainTime, remainBudget, estimatedDensityTraffic, estimatedDensityFee)

        cuttingMethod = 0

        #print(computedReward, computedDensityReward, 'compare')

        '''if computedDensityReward > computedReward:
            pullArms = pullDensityArms
            cuttingMethod = 1'''


        return pullArms
    

    def MultiObjectiveOptimization(self, remainAnswers, estimatedTraffic, estimatedFee):
        omega=para.omega

        objectiveFunc = lambda x: np.dot(omega * estimatedFee, x) + (1 - omega) * np.max(x)

        constraints = ({'type': 'ineq', 'fun': lambda x: np.dot(estimatedTraffic * self.estimatedRewards, x) - remainAnswers})

        bounds = [(0, None) for _ in range(para.nArms)]

        firstGuess = np.ones(para.nArms)

        solution = optimize.minimize(objectiveFunc, firstGuess, method='SLSQP', bounds=bounds,
               constraints=constraints)


        normalizedSolution = solution.x / np.max(solution.x)


        return normalizedSolution
        
    
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




