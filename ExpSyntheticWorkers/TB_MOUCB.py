#time-batch-multi-objective-ucb
import numpy as np
import time
from scipy import optimize
from matplotlib import pyplot as plt
#from DesignBatch import designBatch
from Parameters import para


class TB_MOUCBalgorithm(object):
    def __init__(self, experiment, workerset):
        self.atomPullingCounts = np.zeros(para.nArms[workerset])
        self.atomReward = np.zeros(para.nArms[workerset])
        self.estimatedRewards = np.zeros(para.nArms[workerset])
        self.estimatedTraffic = np.zeros(para.nArms[workerset])
        self.delta = para.delta[experiment]
        self.probs = np.zeros(para.nArms[workerset])
        self.expIndex = experiment
        self.workerset = workerset


    def get_arms(self, remainAnswers, estimatedTraffic, estimatedFee, estimatedDensityTraffic, estimatedDensityFee):
        
        self.estimatedRewards = self.atomReward / self.atomPullingCounts + np.sqrt((self.delta * np.log(np.sum(self.atomPullingCounts))) / self.atomPullingCounts)


        pullArms = self.MultiObjectiveOptimization(remainAnswers, estimatedTraffic, estimatedFee)
        #computedDensityReward, pullDensityArms = self.optimization(remainTime, remainBudget, estimatedDensityTraffic, estimatedDensityFee)

        cuttingMethod = 0

        #print(computedReward, computedDensityReward, 'compare')

        '''if computedDensityReward > computedReward:
            pullArms = pullDensityArms
            cuttingMethod = 1'''


        return pullArms, cuttingMethod
    

    def MultiObjectiveOptimization(self, remainAnswers, estimatedTraffic, estimatedFee):
        omega=para.omega[self.expIndex]

        objectiveFunc = lambda x: np.dot(omega * estimatedFee, x) + (1 - omega) * np.max(x)

        constraints = ({'type': 'ineq', 'fun': lambda x: np.dot(estimatedTraffic * self.estimatedRewards, x) - remainAnswers})

        bounds = [(0, None) for _ in range(para.nArms[self.workerset])]

        firstGuess = np.ones(para.nArms[self.workerset])

        solution = optimize.minimize(objectiveFunc, firstGuess, method='SLSQP', bounds=bounds,
               constraints=constraints)

        #print(solution)

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




