from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from WorkerTraffic import workerTraffic
from NewTrafficFuctions import estimateTraffic
from ComputingGaps import computeGap
from AMTexperimentA.parameter import para

dayM = ['11', '15', '22']
dayS = ['04', '13', '19']

cordModifyT = [[[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-2, 1], [-2, 1], [-2, 1], [-2, 1]], 
               [[-1, 1], [-1, 1], [1.6, 1], [-1, 1], [-2, 10], [-2, 1], [-2, -8], [-2, 3]], 
               [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-2, 1], [-1, 1], [-1, 1], [-1, 4]]]
cordModifyE = [[[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [4, 0], [1, -5], [-1, 1]], 
               [[-1, 1], [-1, 1], [150, 1], [-1, 1], [-1, 1], [1, 1], [-1, -8], [-1, 1]], 
               [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-200, 1], [10, 0], [5, -6], [5, -1]]]

visibleornot = [[0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], 
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]]

meansM = np.array([[[  0.38307892,   0.42269015],
[ -1.01276895,  -1.30367896],
[ 50.19219516, 158.55464911],
[ 16.60843479,  63.18485816]],
[[ 1.04880637e-01,  4.06148354e-01],
[-1.02177779e+00, -8.37110617e-01],
[ 5.00397026e+01,  1.90286153e+02],
[ 1.65666016e+01,  9.47429231e+01]],
[[  0.4862642,    0.45380786],
 [ -1.11127394,  -0.71554844],
 [ 49.76683367, 150.51672602],
 [ 17.06449112,  94.19191207]]])


stdsM = np.array([[[ 0.05285546,  0.10237001],
[ 0.08164421,  0.18824441],
[ 3.27340566, 11.744372  ],
[ 0.87544456,  5.09844762]],
[[0.01782162, 0.07445744],
[0.03986911, 0.0971944 ],
[1.99580375, 9.9324049 ],
[0.6251417,  5.29213437]],
[[ 0.09541437,  0.08278064],
 [ 0.17329609,  0.13795514],
 [ 4.59158896, 10.70244692],
 [ 1.86340169,  5.7496948 ]]
])



meansS = np.array([[[ 1.99030340e-01,  1.84442404e-01],
[-1.37410989e+00, -1.18590439e+00],
[ 9.95691310e+01, 1.98892562e+02],
[ 2.07976325e+01,  5.40888454e+01]],
[[ 1.14406959e-01,  1.54486406e-01],
[-8.66810850e-01, -6.62297248e-01],
[ 5.00550633e+01,  1.89980525e+02],
[ 1.71604051e+01,  9.73553430e+01]],
[[ 1.18833754e-01,  1.12794411e-01],
 [-2.15473120e-01, -3.28123273e-01],
 [ 6.03061068e+01,  1.50323000e+02],
 [ 3.92960830e+01,  9.60449115e+01]]])


stdsS = np.array([[[ 0.01680497,  0.02600571],
[ 0.05699101,  0.0892627 ],
[ 3.97828967, 10.08559845],
[ 0.48364509,  2.000826  ]],
[[0.01206822, 0.00611018],
[0.06610306, 0.02408827],
[3.14988864, 5.23098518],
[0.6252379,  1.04072648]],
[[0.00872362, 0.00607595],
 [0.05138761, 0.03262081],
 [2.98983919, 5.93267253],
 [1.07484783, 2.15491064]]])



for i in range(len(dayM)):
    day1 = dayM[i]
    day2 = dayS[i]


    #CrowdBwO-S1 results
    dfLP = pd.read_csv('AMTexperimentA/exp03-' + day1 + '/realEXPresultsLP' + day1 + '.csv')
    dfLPPara = pd.read_csv('AMTexperimentA/exp03-' + day1 + '/ThroughputandFeeParaLP' + day1 + '.csv', header=None)


    rewardLP1 = np.array(dfLP['2'])
    pulledLP1 = np.array(dfLP['4']).astype(np.int64)
    usedTimeLP1 = np.array(dfLP['6'])

    rewardLP2 = np.array(dfLP['3'])
    pulledLP2 = np.array(dfLP['5']).astype(np.int64)
    usedTimeLP2 = np.array(dfLP['7'])

    rewardLP = rewardLP1 + rewardLP2
    usedBudgetLP = np.array(dfLP['8'])
    timeLP = np.array(dfLP['9'])


    estimatedAccuracyLP1 = rewardLP1[-1] / pulledLP1[-1]
    estimatedAccuracyLP2 = rewardLP2[-1] / pulledLP2[-1]
    estimatedAveThroughputLP1 = pulledLP1[-1] / usedTimeLP1[-1]
    estimatedAveThroughputLP2 = pulledLP2[-1] / usedTimeLP2[-1]

    print('CrowdBwO-S1 Results: ')
    print('P1 accuracy:  ', estimatedAccuracyLP1, '(', pulledLP1[-1], ')', '  P2 accuracy:  ', estimatedAccuracyLP2, '(', pulledLP2[-1], ')', )
    print('P1 throughput:  ', estimatedAveThroughputLP1, '  P2 throughput:  ', estimatedAveThroughputLP2)
    print('P1 used time', usedTimeLP1[-1], 'P2 used time', usedTimeLP2[-1])
    

    #get worker throughput data
    trainingDataLP =  pd.read_csv('AMTexperimentA/exp03-' + day1 + '/timeTrainingDataLP' + day1 + '.csv')

    #use the data to train the worker throughput Models for two worker sets
    '''meansLP1, stdsLP1 = workerTraffic.train(trainingDataLP['time1'], trainingDataLP['workers1'], 0)
    meansLP2, stdsLP2 = workerTraffic.train(trainingDataLP['time2'], trainingDataLP['workers2'], 1)

    meansLP = np.column_stack((meansLP1, meansLP2))
    stdsLP = np.column_stack((stdsLP1, stdsLP2))

    print(meansLP, stdsLP, 'training results')
    '''

    meansLP = meansM[i]
    stdsLP = stdsM[i]


    #find the best priors
    x = np.arange(0, 30, 1)

    y1 = -np.exp(-meansLP[0][0] * x + meansLP[1][0]) * meansLP[2][0] + meansLP[3][0]
    y2 = -np.exp(-meansLP[0][1] * x + meansLP[1][1]) * meansLP[2][1] + meansLP[3][1]


    plt.plot(y1,color='r',  label='LPdel1')
    plt.plot(y2,color='b',  label='LPdel2')
    plt.plot(np.array(trainingDataLP['time1']), np.array(trainingDataLP['workers1']), 'ro', label='actual1')
    plt.plot(np.array(trainingDataLP['time2']), np.array(trainingDataLP['workers2']), 'bo', label='actual2')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    #plt.show()


    #single worker set assignment results
    dfSPLP = pd.read_csv('AMTexperimentB/exp03-' + day2 + '/realEXPresultsSPLP' + day2 + '.csv')

    rewardSPLP1 = np.array(dfSPLP['0'])
    pulledSPLP1 = np.array(dfSPLP['2']).astype(np.int64)
    usedBudgetSPLP1 = np.array(dfSPLP['4'])
    FeasibleSPLP1 = np.array(dfSPLP['7'])
    usedTimeSPLP1 = np.sum(FeasibleSPLP1)

    rewardSPLP2 = np.array(dfSPLP['1'])
    pulledSPLP2 = np.array(dfSPLP['3']).astype(np.int64)
    usedBudgetSPLP2 = np.array(dfSPLP['5'])
    FeasibleSPLP2 = np.array(dfSPLP['8'])
    usedTimeSPLP2 = np.sum(FeasibleSPLP2)


    timeSPLP = np.array(dfSPLP['6'])


    AWIndex = np.where((usedBudgetSPLP1 + usedBudgetSPLP2) <= 1500)[0]
    AWIndex = np.where(timeSPLP[AWIndex] <= 25)[0][-1]
    AWIndex += 1
    FeasibleAWLP = np.ones(AWIndex)
    FeasibleAWLP[0] = 0

    rewardAWLP = (rewardSPLP1 + rewardSPLP2)[:AWIndex]
    pulledAWLP1 = pulledSPLP1[:AWIndex].astype(np.int64)
    pulledAWLP2 = pulledSPLP2[:AWIndex].astype(np.int64)
    usedBudgetAWLP1 = usedBudgetSPLP1[:AWIndex]
    usedBudgetAWLP2 = usedBudgetSPLP2[:AWIndex]
    usedtimeAWLP = np.sum(FeasibleAWLP)


    estimatedAccuracySPLP1 = rewardSPLP1[-1] / pulledSPLP1[-1]
    estimatedAccuracySPLP2 = rewardSPLP2[-1] / pulledSPLP2[-1]
    estimatedAveThroughputSPLP1 = pulledSPLP1[-1] / usedTimeSPLP1
    estimatedAveThroughputSPLP2 = pulledSPLP2[-1] / usedTimeSPLP2


    print('Single Worker Set Results: ')
    print('P1 accuracy:  ', estimatedAccuracySPLP1, '(', pulledSPLP1[-1], ')', '  P2 accuracy:  ', estimatedAccuracySPLP2, '(', pulledSPLP2[-1], ')', )
    print('P1 throughput:  ', estimatedAveThroughputSPLP1, '  P2 throughput:  ', estimatedAveThroughputSPLP2)
    print('P1 used time', usedTimeSPLP1, 'P2 used time', usedTimeSPLP2)
   

    #use the data to train the worker throughput Models for two worker sets
    '''meansSPLP1, stdsSPLP1 = workerTraffic.train((np.array(timeSPLP) * np.array(FeasibleSPLP1))[:int(usedTimeSPLP1)+1], (np.array(pulledSPLP1) * np.array(FeasibleSPLP1))[:int(usedTimeSPLP1)+1], 0)
    meansSPLP2, stdsSPLP2 = workerTraffic.train((np.array(timeSPLP) * np.array(FeasibleSPLP2))[:int(usedTimeSPLP2)+1], (np.array(pulledSPLP2) * np.array(FeasibleSPLP2))[:int(usedTimeSPLP2)+1], 1)

    meansSPLP = np.column_stack((meansSPLP1, meansSPLP2))
    stdsSPLP = np.column_stack((stdsSPLP1, stdsSPLP2))

    print(meansSPLP, stdsSPLP, 'training results')'''

    meansSPLP =  meansS[i]
    stdsSPLP = stdsS[i]

    '''
    df = pd.DataFrame(np.column_stack((estimatedAccuracyLP1.round(2), int(pulledLP1[-1]), estimatedAccuracyLP2.round(2), int(pulledLP2[-1]), estimatedAccuracySPLP1.round(2), int(pulledSPLP1[-1]), estimatedAccuracySPLP2.round(2), int(pulledSPLP2[-1]))))
    df.to_csv('./resultValues/resultsAccS1.csv', mode='a', index=False)
    
    df = pd.DataFrame(np.column_stack((meansLP.round(2), meansSPLP.round(2))))
    df.to_csv('./resultValues/resultsWorkersS1.csv', mode='a', index=False)
    '''

    x = np.arange(0, 30, 1)

    y1 = -np.exp(-meansSPLP[0][0] * x + meansSPLP[1][0]) * meansSPLP[2][0] + meansSPLP[3][0]
    y2 = -np.exp(-meansSPLP[0][1] * x + meansSPLP[1][1]) * meansSPLP[2][1] + meansSPLP[3][1]


    plt.plot(y1, color='r', label='Model1')
    plt.plot(y2, color='b', label='Model2')
    plt.plot(np.array(timeSPLP) * np.array(FeasibleSPLP1), np.array(pulledSPLP1) * np.array(FeasibleSPLP1), 'ro', label='Actual1')
    plt.plot(np.array(timeSPLP) * np.array(FeasibleSPLP2), np.array(pulledSPLP2) * np.array(FeasibleSPLP2), 'bo', label='Actual2')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    #plt.show()

    #Models of single worker set utilization with optimal publish windows
    #estimatedAccuracy1 = (rewardLP1[-1] + rewardSPLP1[-1]) / (pulledLP1[-1] + pulledSPLP1[-1])
    #estimatedAccuracy2 = (rewardLP2[-1] + rewardSPLP2[-1]) / (pulledLP2[-1] + pulledSPLP2[-1])


    optRewardTraceLP, rewardLP, optFeeTraceLP, usedBudgetLP, gapLP, timeTraceLP = computeGap(1, estimatedAccuracyLP1, estimatedAccuracyLP2, timeLP, rewardLP, usedBudgetLP, meansLP, stdsLP, i)
    optRewardTraceSPLP1, rewardSPLP1, optFeeTraceSPLP1, usedBudgetSPLP1, gapSPLP1, timeTraceSPLP1 = computeGap(1, estimatedAccuracySPLP1, estimatedAccuracySPLP2, np.cumsum(FeasibleSPLP1), rewardSPLP1, usedBudgetSPLP1, meansSPLP, stdsSPLP, i)
    optRewardTraceSPLP2, rewardSPLP2, optFeeTraceSPLP2, usedBudgetSPLP2, gapSPLP2, timeTraceSPLP2 = computeGap(1, estimatedAccuracySPLP1, estimatedAccuracySPLP2, np.cumsum(FeasibleSPLP1), rewardSPLP2, usedBudgetSPLP2, meansSPLP, stdsSPLP, i)
    optRewardTraceAWLP, rewardAWLP, optFeeTraceAWLP, usedBudgetAWLP, gapAWLP, timeTraceAWLP = computeGap(1, estimatedAccuracySPLP1, estimatedAccuracySPLP2, np.cumsum(FeasibleAWLP), rewardAWLP, usedBudgetAWLP1+usedBudgetAWLP2, meansSPLP, stdsSPLP, i)
    



    #------------plot---------------------------------
    plt.rcParams.update({'font.size': 22})


    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(timeTraceLP, optRewardTraceLP, label='Optimal')
    plt.plot(timeTraceLP, rewardLP, marker='o', label='CrowdBwO-S1')
    plt.axvline(x=para.timeLimit, color='r', linestyle=':', label='Time\nBudget')
    plt.xlabel('Time Utilization (Minute)')
    plt.ylabel('#High-quality Results')
    #plt.tight_layout()
    #plt.title('High-quality Results of Acquired by CrowdBwO-S1 (Task 1)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    fname = './resultFiguresReal/qualityResultsbyTimeMulti' + day1 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()


    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(timeTraceSPLP1, optRewardTraceSPLP1, label='Optimal')
    plt.plot(timeTraceSPLP1, rewardSPLP1, label='WSW-S1\n(AMT(<500))')
    plt.plot(timeTraceSPLP2, rewardSPLP2, label='OSW-S1\n(AMT(≥500))')
    plt.plot(timeTraceAWLP, rewardAWLP, label='AW-S1')
    plt.axvline(x=para.timeLimit, color='r', linestyle=':', label='Time\nBudget')
    plt.xlabel('Time Utilization (Minute)')
    plt.ylabel('#High-quality Results')
    #plt.tight_layout()
    #plt.title('High-quality Results of Acquired by Single Worker Set Method (Task 1)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    fname = './resultFiguresReal/qualityResultsbyTimeSingle' + day2 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()



    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(optFeeTraceLP, optRewardTraceLP, label='Optimal')
    plt.plot(usedBudgetLP, rewardLP, marker='o', label='CrowdBwO-S1')
    plt.axvline(x=para.budget, color='r', linestyle=':', label='Expense\nBudget')
    plt.xlabel('Accumulated Expense ($0.01)')
    plt.ylabel('#High-quality Results')
    #plt.tight_layout()
    #plt.title('Highquality Results of Acquired by CrowdBwO-S1 (Task 1)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    fname = './resultFiguresReal/qualityResultsbyExpenseMulti' + day1 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()


    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(optFeeTraceSPLP1, optRewardTraceSPLP1, label='Optimal')
    plt.plot(usedBudgetSPLP1, rewardSPLP1, label='WSW-S1\n(AMT(<500))')
    plt.plot(usedBudgetSPLP2, rewardSPLP2, label='OSW-S1\n(AMT(≥500))')
    plt.plot(usedBudgetAWLP, rewardAWLP, label='AW-S1')
    plt.axvline(x=para.budget, color='r', linestyle=':', label='Expense\nBudget')
    plt.xlabel('Accumulated Expense ($0.01)')
    plt.ylabel('#High-quality Results')
    #plt.tight_layout()
    #plt.title('High-quality Results of Acquired by Single Worker Set Method (Task 1)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    fname = './resultFiguresReal/qualityResultsbyExpenseSingle' + day2 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()



    #-----------------------loss--------------------
    maxLPIndex = np.argmax(gapLP)
    maxSPLP1Index = np.argmax(gapSPLP1)
    maxSPLP2Index = np.argmax(gapSPLP2)
    maxAWLPIndex = np.argmax(gapAWLP)

    #print(gapLP, maxLPIndex, timeTraceLP[maxLPIndex], gapLP[maxLPIndex]+1, 'test')
    
    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(timeTraceLP, gapLP, marker='o', label='CrowdBwO-S1')
    plt.plot(timeTraceSPLP1, gapSPLP1, marker='o', label='WSW-S1\n(AMT(<500))')
    plt.plot(timeTraceSPLP2, gapSPLP2, marker='o', label='OSW-S1\n(AMT(≥500))')
    plt.plot(timeTraceAWLP, gapAWLP, marker='o', label='AW-S1')
   
    plt.text(timeTraceLP[maxLPIndex]+cordModifyT[i][0][0], gapLP[maxLPIndex]+cordModifyT[i][0][1], str(rewardLP[maxLPIndex]) + '/' +str(optRewardTraceLP[maxLPIndex]), color='#66B2FF', fontweight='semibold', visible=visibleornot[i][0], fontsize=15)
    plt.text(timeTraceSPLP1[maxSPLP1Index]+cordModifyT[i][1][0], gapSPLP1[maxSPLP1Index]+cordModifyT[i][1][1], str(rewardSPLP1[maxSPLP1Index]) + '/' +str(optRewardTraceSPLP1[maxSPLP1Index]), color='#FF8000', fontweight='semibold', visible=visibleornot[i][1], fontsize=15)
    plt.text(timeTraceSPLP2[maxSPLP2Index]+cordModifyT[i][2][0], gapSPLP2[maxSPLP2Index]+cordModifyT[i][2][1], str(rewardSPLP2[maxSPLP2Index]) + '/' +str(optRewardTraceSPLP2[maxSPLP2Index]), color='#00CC00', fontweight='semibold', visible=visibleornot[i][2], fontsize=15)
    plt.text(timeTraceAWLP[maxAWLPIndex]+cordModifyT[i][3][0], gapAWLP[maxAWLPIndex]+cordModifyT[i][3][1], str(rewardAWLP[maxAWLPIndex]) + '/' +str(optRewardTraceAWLP[maxAWLPIndex]), color='#FF6666', fontweight='semibold', visible=visibleornot[i][3], fontsize=15)
    
    plt.text(timeTraceLP[-1]+cordModifyT[i][4][0], gapLP[-1]+cordModifyT[i][4][1], str(rewardLP[-1]) + '/' +str(optRewardTraceLP[-1]), color='#66B2FF', fontweight='semibold', visible=visibleornot[i][4], fontsize=15)
    plt.text(timeTraceSPLP1[-1]+cordModifyT[i][5][0], gapSPLP1[-1]+cordModifyT[i][5][1], str(rewardSPLP1[-1]) + '/' +str(optRewardTraceSPLP1[-1]), color='#FF8000', fontweight='semibold', visible=visibleornot[i][5], fontsize=15)
    plt.text(timeTraceSPLP2[-1]+cordModifyT[i][6][0], gapSPLP2[-1]+cordModifyT[i][6][1], str(rewardSPLP2[-1]) + '/' +str(optRewardTraceSPLP2[-1]), color='#00CC00', fontweight='semibold', visible=visibleornot[i][6], fontsize=15)
    plt.text(timeTraceAWLP[-1]+cordModifyT[i][7][0], gapAWLP[-1]+cordModifyT[i][7][1], str(rewardAWLP[-1]) + '/' +str(optRewardTraceAWLP[-1]), color='#FF6666', fontweight='semibold', visible=visibleornot[i][7], fontsize=15)
   
    plt.axvline(x=para.timeLimit, color='r', linestyle=':', label='Time\nBudget')
    plt.xlabel('Time Utilization (Minute)')
    plt.ylabel('Percentage of the\nOptimal #High-\nquality Results (%)')
    #plt.tight_layout()
    #plt.title('High-quality Results Comparison on Task 1')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    fname = './resultFiguresReal/qualityResultsLossbyTime' + day1 + day2 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()
    

    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(usedBudgetLP, gapLP, marker='o', label='CrowdBwO-S1')
    plt.plot(usedBudgetSPLP1, gapSPLP1, marker='o', label='WSW-S1\n(AMT(<500))')
    plt.plot(usedBudgetSPLP2, gapSPLP2, marker='o', label='OSW-S1\n(AMT(≥500))')
    plt.plot(usedBudgetAWLP, gapAWLP, marker='o', label='AW-S1')
    
    plt.text(usedBudgetLP[maxLPIndex]+cordModifyE[i][0][0], gapLP[maxLPIndex]+cordModifyE[i][0][1], str(rewardLP[maxLPIndex]) + '/' +str(optRewardTraceLP[maxLPIndex]), color='#66B2FF', fontweight='semibold', visible=visibleornot[i][8], fontsize=15)
    plt.text(usedBudgetSPLP1[maxSPLP1Index]+cordModifyE[i][1][0], gapSPLP1[maxSPLP1Index]+cordModifyE[i][1][1], str(rewardSPLP1[maxSPLP1Index]) + '/' +str(optRewardTraceSPLP1[maxSPLP1Index]), color='#FF8000', fontweight='semibold', visible=visibleornot[i][9], fontsize=15)
    plt.text(usedBudgetSPLP2[maxSPLP2Index]+cordModifyE[i][2][0], gapSPLP2[maxSPLP2Index]+cordModifyE[i][2][1], str(rewardSPLP2[maxSPLP2Index]) + '/' +str(optRewardTraceSPLP2[maxSPLP2Index]), color='#00CC00', fontweight='semibold', visible=visibleornot[i][10], fontsize=15)
    plt.text(usedBudgetAWLP[maxAWLPIndex]+cordModifyE[i][3][0], gapAWLP[maxAWLPIndex]+cordModifyE[i][3][1], str(rewardAWLP[maxAWLPIndex]) + '/' +str(optRewardTraceAWLP[maxAWLPIndex]), color='#FF6666', fontweight='semibold', visible=visibleornot[i][11], fontsize=15)
    
    plt.text(usedBudgetLP[-1]+cordModifyE[i][4][0], gapLP[-1]+cordModifyE[i][4][1], str(rewardLP[-1]) + '/' +str(optRewardTraceLP[-1]), color='#66B2FF', fontweight='semibold', visible=visibleornot[i][12], fontsize=15)
    plt.text(usedBudgetSPLP1[-1]+cordModifyE[i][5][0], gapSPLP1[-1]+cordModifyE[i][5][1], str(rewardSPLP1[-1]) + '/' +str(optRewardTraceSPLP1[-1]), color='#FF8000', fontweight='semibold', visible=visibleornot[i][13], fontsize=15)
    plt.text(usedBudgetSPLP2[-1]+cordModifyE[i][6][0], gapSPLP2[-1]+cordModifyE[i][6][1], str(rewardSPLP2[-1]) + '/' +str(optRewardTraceSPLP2[-1]), color='#00CC00', fontweight='semibold', visible=visibleornot[i][14], fontsize=15)
    plt.text(usedBudgetAWLP[-1]+cordModifyE[i][7][0], gapAWLP[-1]+cordModifyE[i][7][1], str(rewardAWLP[-1]) + '/' +str(optRewardTraceAWLP[-1]), color='#FF6666', fontweight='semibold', visible=visibleornot[i][15], fontsize=15)
    
    plt.axvline(x=para.budget, color='r', linestyle=':', label='Expense\nBudget')
    plt.xlabel('Accumulative Expense ($0.01)')
    plt.ylabel('Percentage of the\nOptimal #High-\nquality Results (%)')
    #plt.tight_layout()
    #plt.title('High-quality Results Comparison on Task 1')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    fname = './resultFiguresReal/qualityResultsLossbyExpense' + day1 + day2 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()


    plt.close('all')





