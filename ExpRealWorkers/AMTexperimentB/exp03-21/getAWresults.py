import pandas as pd
import numpy as np
from datetime import datetime


feePara = np.array([3, 3]) #the incentive is $0.02, fee to platform is $0.01

def feeCount(Workers):
    fee = Workers * feePara

    fee[Workers == 0] = 0
    
    return np.sum(fee)


results = pd.read_csv('./exp03-21/AssignmentResults22.csv')


ans = np.array(results['answer0'])
score = np.array(results['score'])
time = np.array(results['submitTime'])

for i in range(len(time)):
    splits = time[i].split(' ')[1]
    splits = splits.split('+')[0]
    splits = splits.split(':')
    if int(splits[0]) < 17:
        time[i] = (int(splits[0]) + 7) * 60 + int(splits[1]) - 15

    elif int(splits[0]) >= 17:
        time[i] = (int(splits[0]) - 17) * 60 + int(splits[1]) - 15




nArms = 2
rewardsR = np.zeros(nArms)
pulledCountsR = np.zeros(nArms)
Budget1R = [[0]]
Budget2R = [[0]]
tR = [[0]]
PullingisFeasible = np.ones(nArms)
PullingisFeasibleR = np.zeros(nArms)
finalRewards = np.zeros(nArms)
finalPulls = np.zeros(nArms)


for i in range(1, time[-1]+1):
    collectedAnswers = np.zeros(nArms)
    correctAnswers = np.zeros(nArms)

    for r in range(len(ans)):
        if time[r] <= i:
            if score[r] == 'stask1':
                collectedAnswers[0] += 1
                if str(ans[r]) == '350':
                    correctAnswers[0] += 1


            elif score[r] == 'stask2':
                collectedAnswers[1] += 1
                if str(ans[r]) == '350':
                    correctAnswers[1] += 1

    

    qualified = collectedAnswers * np.array([1, 0])
    nonqualified = collectedAnswers * np.array([0, 1])

    usedBudget1 = feeCount(qualified)
    usedBudget2 = feeCount(nonqualified)

    rewardsR = np.vstack((rewardsR, correctAnswers))
    pulledCountsR = np.vstack((pulledCountsR, collectedAnswers))
    Budget1R.append([usedBudget1])
    Budget2R.append([usedBudget2])
    tR.append([i])
    PullingisFeasibleR = np.vstack((PullingisFeasibleR, PullingisFeasible))

    if np.sum(correctAnswers) >= 40:
        PullingisFeasible *= 0




results = np.hstack((rewardsR, pulledCountsR, np.array(Budget1R), np.array(Budget2R), np.array(tR), PullingisFeasibleR))
df = pd.DataFrame(results)

df.to_csv('./exp03-21/realEXPresultsAWMO21.csv', mode='w')