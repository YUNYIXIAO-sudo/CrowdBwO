from fee import feeCount
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt
from parameter import para
from datetime import datetime

'''
from hit1 import publishHIT1
from hit2 import publishHIT2
from terminateHIT import stopHIT
from answers import analyzeResults   
'''


def publishHIT1(qua):
    return 1

def publishHIT2(qua):
    return 2

def stopHIT(id):
    ids = id

def analyzeResults(time, qualification):
    collectedAnswers = np.sqrt(np.array([10, 70]) * time).astype(np.int64)
    correctAnswers = np.sqrt(np.array([6, 60]) * time).astype(np.int64)

    return collectedAnswers, correctAnswers
#remember to delete the time !!!!!!!


#mturk '35OXBT565ER91QGMB3F119JHKAFV2Q'
qualification = '332K4KOFDLOGBZDM5VBUQVISG7S0DA'

hitid1 = publishHIT1(qualification)
hitid2 = publishHIT2(qualification)


B = para.budget
rewardsR = np.zeros(para.nArms)
pulledCountsR = np.zeros(para.nArms)
Budget1R = [[0]]
Budget2R = [[0]]
tR = [[0]]
PullingisFeasible = np.ones(para.nArms)
PullingisFeasibleR = np.zeros(para.nArms)
t = 0


while np.any(PullingisFeasible == 1) == 1:
    #time.sleep(60)
    t += 1
    pulled, reward = analyzeResults(t, qualification)
    qualified = pulled * np.array([1, 0])
    nonqualified = pulled * np.array([0, 1])

    usedBudget1 = feeCount(qualified)
    usedBudget2 = feeCount(nonqualified)

    rewardsR = np.vstack((rewardsR, reward))
    pulledCountsR = np.vstack((pulledCountsR, pulled))
    Budget1R.append([usedBudget1])
    Budget2R.append([usedBudget2])
    tR.append([t])
    PullingisFeasibleR = np.vstack((PullingisFeasibleR, PullingisFeasible))

    
    print('RESULT OF A BATCH    t: ' + str(t) + ' assign: ' + str(pulled) + ' fee1: ' + str(usedBudget1 / 100) + ' fee2: ' + str(usedBudget2 / 100))
                    

    if (t >= para.timeLimit or usedBudget1 >= B) and PullingisFeasible[0] == 1:
        stopHIT(hitid1)
        PullingisFeasible[0] = 0
    if (t >= para.timeLimit or usedBudget2 >= B) and PullingisFeasible[1] == 1:
        stopHIT(hitid2)
        PullingisFeasible[1] = 0
    
    
results = np.hstack((rewardsR, pulledCountsR, np.array(Budget1R), np.array(Budget2R), np.array(tR), PullingisFeasibleR))
df = pd.DataFrame(results)
#, columns=['Selected Project', 'Cumulative Rewards', 'Number of Assignments', 'Remaining Budget', 'Spent Time'])

df.to_csv('realEXPresultsSPLP' + str(datetime.now().day) + '.csv', mode='w')

parameterFile = open('parameter.py', mode='r')
parameterNote = parameterFile.read()
file = open('note' + str(datetime.now().day) + '.txt', mode='w')
file.write(parameterNote)
file.close()