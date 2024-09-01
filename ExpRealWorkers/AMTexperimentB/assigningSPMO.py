from fee import feeCount
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt
from parameter import para
from datetime import datetime


from hitApproval1 import publishHIT1
from hitApproval2 import publishHIT2
from terminateHIT import stopHIT
from answers import analyzeResults   

#remember to delete the time !!!!!!!

timelimitforsetting2 = 720
qualification = '32R8QD8BQAW05R82HB4D70MWWUQDCK'

hitid1 = publishHIT1(qualification)
hitid2 = publishHIT2(qualification)


rewardsR = np.zeros(para.nArms)
pulledCountsR = np.zeros(para.nArms)
Budget1R = [[0]]
Budget2R = [[0]]
tR = [[0]]
PullingisFeasible = np.ones(para.nArms)
PullingisFeasibleR = np.zeros(para.nArms)
t = 0


while np.any(PullingisFeasible == 1) == 1:
    time.sleep(60)
    t += 1
    pulled, reward = analyzeResults(qualification)
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
        
    if (reward[0] >= para.targetAnswers and PullingisFeasible[0] == 1) or t >= timelimitforsetting2:
        stopHIT(hitid1)
        PullingisFeasible[0] = 0
    if (reward[1] >= para.targetAnswers and PullingisFeasible[1] == 1) or t >= timelimitforsetting2:
        stopHIT(hitid2)
        PullingisFeasible[1] = 0
        
    
results = np.hstack((rewardsR, pulledCountsR, np.array(Budget1R), np.array(Budget2R), np.array(tR), PullingisFeasibleR))
df = pd.DataFrame(results)
#, columns=['Selected Project', 'Cumulative Rewards', 'Number of Assignments', 'Remaining Budget', 'Spent Time'])

df.to_csv('realEXPresultsSPMO' + str(datetime.now().day) + '.csv', mode='w')


parameterFile = open('parameter.py', mode='r')
parameterNote = parameterFile.read()
file = open('note' + str(datetime.now().day) + '.txt', mode='w')
file.write(parameterNote)
file.close()