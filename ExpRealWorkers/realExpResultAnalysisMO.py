from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from WorkerTraffic import workerTraffic
from ComputingGaps import computeGap
from AMTexperimentA.parameter import para


target = [50, 30, 40]
expValue = [2000, 6000, 7000]

dayM = ['12', '14', '20']
dayS = ['05', '18', '21']
ticks = [85, 65, 55]

cordModify = [[[-14, 0.1], [1, -0.1], [1, -0.1], [1, -0.4], [-2, 0.3], [-2, 0.3], [-2, 0.3], [-2, 0.3]], 
               [[-8, 0.2], [1, -0.1], [1, -0.15], [-1.5, 0.1], [-2, 1], [-5, 0.1], [-2, -1], [-2, 1]], 
               [[-4, -0.4], [-5, 0.2], [-4, 0.1], [0.5, -0.2], [-1, 1], [-5, 0.1], [-1, 1], [-1, 1]]]

visibleornot = [[1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 1, 0, 0],
                [1, 1, 1, 1, 0, 1, 0, 0]]


meansM = np.array([[[  0.49545116,   0.51217253],
[ -0.97394264,  -1.19910697],
[ 49.79621045, 148.64361117],
[ 17.99104605,  60.87362004]],
[[  0.52560456,   0.48735042],
[ -1.14052096,  -0.84599654],
[ 49.35513502, 130.32753585],
[ 21.57596912,  74.98861416]],
[[  0.51126244,   0.48798017],
 [ -1.10788906,  -0.73649552],
 [ 49.53355958, 130.62313811],
 [ 19.89058603,  85.54184038]] 
])

stdsM = np.array([[[ 0.10548278,  0.0998392 ],
[ 0.34070697,  0.37018133],
[ 5.08726194, 11.96773876],
[ 3.34457008,  9.11341674]],
[[ 0.10194195,  0.11009831],
[ 0.32809142,  0.43482393],
[ 5.02477221, 12.11312335],
[ 2.99060474, 11.01856819]],
[[ 0.10895248,  0.10661534],
[ 0.37262299,  0.33096008],
[ 5.15600662, 12.28945868],
[ 3.74682772,  9.60451688]]])


meansS = np.array([[[ 7.39591017e-02,  9.62428287e-02],
[-9.90973056e-01, -9.03873112e-01],
[ 1.19645618e+02,  2.49184033e+02],
[ 4.86483545e+01,  9.70151546e+01]],
[[ 5.36073986e-02,  3.20943402e-01],
[-4.07591951e+00, -3.98865365e-01],
[ 2.99488192e+03,  1.31529209e+02],
[ 5.16540534e+01,  7.29649199e+01]],
[[ 8.15804918e-02,  3.20416706e-01],
 [-4.63730725e+00, -5.76075216e-01],
 [ 2.99643833e+03,  1.50521586e+02],
 [ 4.22121395e+01,  9.09424358e+01]]
])

stdsS = np.array([[[3.39164690e-02, 3.40548927e-03],
[1.04357504e-01, 2.19403323e-02],
[3.73182062e+00, 6.12943598e+00],
[5.70767262e-01, 1.08817736e+00]],
[[4.03934008e-03, 6.27866204e-02],
[6.27970124e-02, 1.29357039e-01],
[9.79563325e+00, 9.45409107e+00],
[1.57961410e+00, 5.58332326e+00]],
[[5.05021567e-03, 5.86785268e-02],
 [7.54795665e-02, 1.11780355e-01],
 [1.00442850e+01, 1.00271583e+01],
 [1.30927516e+00, 4.18184816e+00]]
])



for i in range(len(dayM)):

    day1 = dayM[i]
    day2 = dayS[i]


    dfMO = pd.read_csv('AMTexperimentA/exp03-' + day1 + '/realEXPresultsMO' + day1 + '.csv')
    dfMOPara = pd.read_csv('AMTexperimentA/exp03-' + day1 + '/ThroughputandFeeParaMO' + day1 + '.csv', header=None)

    #proposed method results
    rewardMO1 = np.array(dfMO['2'])
    pulledMO1 = np.array(dfMO['4']).astype(np.int64)
    usedTimeMO1 = np.array(dfMO['6'])

    rewardMO2 = np.array(dfMO['3'])
    pulledMO2 = np.array(dfMO['5'])
    usedTimeMO2 = np.array(dfMO['7'])

    rewardMO = rewardMO1 + rewardMO2
    usedBudgetMO = np.array(dfMO['8'])
    timeMO = np.array(dfMO['9'])


    estimatedAccuracyMO1 = rewardMO1[-1] / pulledMO1[-1]
    estimatedAccuracyMO2 = rewardMO2[-1] / pulledMO2[-1]
    estimatedAveThroughputMO1 = pulledMO1[-1] / usedTimeMO1[-1]
    estimatedAveThroughputMO2 = pulledMO2[-1] / usedTimeMO2[-1]

    print('Proposed Method Results: ')
    print('P1 accuracy:  ', estimatedAccuracyMO1, '(', pulledMO1[-1], ')',  '  P2 accuracy:  ', estimatedAccuracyMO2, '(', pulledMO2[-1], ')' )
    print('P1 throughput:  ', estimatedAveThroughputMO1, '  P2 throughput:  ', estimatedAveThroughputMO2)
    print('P1 used time', usedTimeMO1[-1], 'P2 used time', usedTimeMO2[-1])

    
    #get worker throughput data
    trainingDataMO =  pd.read_csv('AMTexperimentA/exp03-' + day1 + '/timeTrainingDataMO' + day1 + '.csv')

    #use the data to train the worker throughput models for two worker sets
    '''meansMO1, stdsMO1 = workerTraffic.train(trainingDataMO['time1'], trainingDataMO['workers1'], 0)
    meansMO2, stdsMO2 = workerTraffic.train(trainingDataMO['time2'], trainingDataMO['workers2'], 1)

    meansMO = np.column_stack((meansMO1, meansMO2))
    stdsMO = np.column_stack((stdsMO1, stdsMO2))

    print(meansMO, stdsMO, 'training results')'''


    meansMO = meansM[i]
    stdsMO = stdsM[i]



    #find the best priors
    x = np.arange(0, 30, 1)

    y1 = -np.exp(-meansMO[0][0] * x + meansMO[1][0]) * meansMO[2][0] + meansMO[3][0]
    y2 = -np.exp(-meansMO[0][1] * x + meansMO[1][1]) * meansMO[2][1] + meansMO[3][1]

    '''
    plt.plot(y1, color='r', label='model1')
    plt.plot(y2, color='b', label='model2')
    plt.plot(np.array(trainingDataMO['time1']), np.array(trainingDataMO['workers1']), 'ro', label='actual1')
    plt.plot(np.array(trainingDataMO['time2']), np.array(trainingDataMO['workers2']), 'bo', label='actual2')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    #plt.show()'''




    #single worker set assignment results
    dfSPMO = pd.read_csv('AMTexperimentB/exp03-' + day2 + '/realEXPresultsSPMO' + day2 + '.csv')

    rewardSPMO1 = np.array(dfSPMO['0'])
    pulledSPMO1 = np.array(dfSPMO['2']).astype(np.int64)
    usedBudgetSPMO1 = np.array(dfSPMO['4'])
    FeasibleSPMO1 = np.array(dfSPMO['7'])
    usedTimeSPMO1 = np.sum(FeasibleSPMO1)

    rewardSPMO2 = np.array(dfSPMO['1'])
    pulledSPMO2 = np.array(dfSPMO['3']).astype(np.int64)
    usedBudgetSPMO2 = np.array(dfSPMO['5'])
    FeasibleSPMO2 = np.array(dfSPMO['8'])
    usedTimeSPMO2 = np.sum(FeasibleSPMO2)


    timeSPMO = np.array(dfSPMO['6'])

    AWIndex = np.where((rewardSPMO1 + rewardSPMO2) <= target[i])[0][-1]
    AWIndex += 1
    FeasibleAWMO = np.ones(AWIndex)
    FeasibleAWMO[0] = 0
    
    rewardAWMO = (rewardSPMO1 + rewardSPMO2)[:AWIndex]
    pulledAWMO1 = pulledSPMO1[:AWIndex].astype(np.int64)
    pulledAWMO2 = pulledSPMO2[:AWIndex].astype(np.int64)
    usedBudgetAWMO1 = usedBudgetSPMO1[:AWIndex]
    usedBudgetAWMO2 = usedBudgetSPMO2[:AWIndex]
    usedtimeAWMO = np.sum(FeasibleAWMO)
    

    estimatedAccuracySPMO1 = rewardSPMO1[-1] / pulledSPMO1[-1]
    estimatedAccuracySPMO2 = rewardSPMO2[-1] / pulledSPMO2[-1]
    estimatedAveThroughputSPMO1 = pulledSPMO1[-1] / usedTimeSPMO1
    estimatedAveThroughputSPMO2 = pulledSPMO2[-1] / usedTimeSPMO2


    print('Single Worker Set Results: ')
    print('P1 accuracy:  ', estimatedAccuracySPMO1, '(', pulledSPMO1[-1], ')',  '  P2 accuracy:  ', estimatedAccuracySPMO2, '(', pulledSPMO2[-1], ')', )
    print('P1 throughput:  ', estimatedAveThroughputSPMO1, '  P2 throughput:  ', estimatedAveThroughputSPMO2)
    print('P1 used time', usedTimeSPMO1, 'P2 used time', usedTimeSPMO2)

   

    #use the data to train the worker throughput models for two worker sets
    '''meansSPMO1, stdsSPMO1 = workerTraffic.train((np.array(timeSPMO) * np.array(FeasibleSPMO1))[:int(usedTimeSPMO1)+1], (np.array(pulledSPMO1) * np.array(FeasibleSPMO1))[:int(usedTimeSPMO1)+1], 0)
    meansSPMO2, stdsSPMO2 = workerTraffic.train((np.array(timeSPMO) * np.array(FeasibleSPMO2))[:int(usedTimeSPMO2)+1], (np.array(pulledSPMO2) * np.array(FeasibleSPMO2))[:int(usedTimeSPMO2)+1], 1)

    meansSPMO = np.column_stack((meansSPMO1, meansSPMO2))
    stdsSPMO = np.column_stack((stdsSPMO1, stdsSPMO2))

    print(meansSPMO, stdsSPMO, 'training results')'''


    
    meansSPMO = meansS[i]
    stdsSPMO = stdsS[i]

    '''
    df = pd.DataFrame(np.column_stack((estimatedAccuracyMO1.round(2), int(pulledMO1[-1]), estimatedAccuracyMO2.round(2), int(pulledMO2[-1]), estimatedAccuracySPMO1.round(2), int(pulledSPMO1[-1]), estimatedAccuracySPMO2.round(2), int(pulledSPMO2[-1]))))
    df.to_csv('./resultValues/resultsAccS2.csv', mode='a', index=False)

    df = pd.DataFrame(np.column_stack((meansMO.round(2), meansSPMO.round(2))))
    df.to_csv('./resultValues/resultsWorkersS2.csv', mode='a', index=False)
    '''

    x = np.arange(0, 150, 1)


    y1 = -np.exp(-meansSPMO[0][0] * x + meansSPMO[1][0]) * meansSPMO[2][0] + meansSPMO[3][0]
    y2 = -np.exp(-meansSPMO[0][1] * x + meansSPMO[1][1]) * meansSPMO[2][1] + meansSPMO[3][1]

    plt.figure(figsize=(8, 4), constrained_layout=True)
    plt.plot(y1, color='r', label='Model')
    #plt.plot(y2, color='b', label='model2')
    plt.plot((np.array(timeSPMO) * np.array(FeasibleSPMO1))[:150], (np.array(pulledSPMO1) * np.array(FeasibleSPMO1))[:150], 'ro', label='Actual')
    #plt.plot(np.array(timeSPMO) * np.array(FeasibleSPMO2), np.array(pulledSPMO2) * np.array(FeasibleSPMO2), 'bo', label='actual2')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    #plt.show()
    plt.savefig('./resultFiguresReal/RealDataPlot' + day2 + '.png', bbox_inches=None, pad_inches=None)


    estimatedAccuracy1 = (rewardMO1[-1] + rewardSPMO1[-1]) / (pulledMO1[-1] + pulledSPMO1[-1])
    estimatedAccuracy2 = (rewardMO2[-1] + rewardSPMO2[-1]) / (pulledMO2[-1] + pulledSPMO2[-1])

    optTimeTraceMO, timeTraceMO, optFeeTraceMO, usedBudgetMO, gapTimeMO, gapFeeMO, gapCombiMO, rewardMO = computeGap(2, estimatedAccuracyMO1, estimatedAccuracyMO2, timeMO, rewardMO, usedBudgetMO, meansMO, stdsMO, i)
    optTimeTraceSPMO1, timeTraceSPMO1, optFeeTraceSPMO1, usedBudgetSPMO1, gapTimeSPMO1, gapFeeSPMO1, gapCombiSPMO1, rewardSPMO1 = computeGap(2, estimatedAccuracySPMO1, estimatedAccuracySPMO2, np.cumsum(FeasibleSPMO1), rewardSPMO1, usedBudgetSPMO1, meansSPMO, stdsSPMO, i)
    optTimeTraceSPMO2, timeTraceSPMO2, optFeeTraceSPMO2, usedBudgetSPMO2, gapTimeSPMO2, gapFeeSPMO2, gapCombiSPMO2, rewardSPMO2 = computeGap(2, estimatedAccuracySPMO1, estimatedAccuracySPMO2, np.cumsum(FeasibleSPMO2), rewardSPMO2, usedBudgetSPMO2, meansSPMO, stdsSPMO, i)
    optTimeTraceAWMO, timeTraceAWMO, optFeeTraceAWMO, usedBudgetAWMO, gapTimeAWMO, gapFeeAWMO, gapCombiAWMO, rewardAWMO = computeGap(2, estimatedAccuracySPMO1, estimatedAccuracySPMO2, np.cumsum(FeasibleAWMO), rewardAWMO, usedBudgetAWMO1+usedBudgetAWMO2, meansSPMO, stdsSPMO, i)
   



    #------------plot---------------------------------
    plt.rcParams.update({'font.size': 22})


    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(rewardMO, optFeeTraceMO, label='Optimal (Proposed Method)')
    plt.plot(rewardMO, usedBudgetMO, marker='o', label='Proposed Method')
    plt.axvline(x = target[i], color='r', linestyle=':', label='#Required Number of \nHigh-quality Results')
    plt.xlabel('#High-quality Results')
    plt.ylabel('Accumulated Expense ($0.01)')
    #plt.tight_layout()
    #plt.title('Accumulated Expense of Proposed Method on Task 1')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    fname = './resultFiguresReal/expensebyHighqualityResultsMulti' + day1 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()


    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(rewardSPMO1, optFeeTraceSPMO1, label='Optimal (Single)')
    plt.plot(rewardSPMO1, usedBudgetSPMO1, label='WSW-S2\n(AMT(<500))')
    plt.plot(rewardSPMO2, usedBudgetSPMO2, label='OSW-S2\n(AMT(≥500))')
    plt.plot(rewardAWMO, usedBudgetAWMO, label='AW-S2')
    plt.axvline(x = target[i], color='r', linestyle=':', label='#Required Number of \nHigh-quality Results')
    plt.xlabel('#High-quality Results')
    plt.ylabel('Accumulated Expense ($0.01)')
    #plt.tight_layout()
    #plt.title('Accumulated Expense of Single Worker Set Method on Task 1')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    fname = './resultFiguresReal/expensebyHighqualityResultsSingle' + day2 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()


    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(rewardMO, optTimeTraceMO, label='Optimal (Proposed Method)')
    plt.plot(rewardMO, timeTraceMO, marker='o', label='Proposed Method')
    plt.axvline(x = target[i], color='r', linestyle=':', label='#Required Number of \nHigh-quality Results')
    plt.xlabel('#High-quality Results')
    plt.ylabel('Time Utilization (Minute)')
    #plt.tight_layout()
    #plt.title('Time Utilization of Proposed Method on Task 1')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    fname = './resultFiguresReal/timebyHighqualityResultsMulti' + day1 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()


    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(rewardSPMO1, optTimeTraceSPMO1, label='Optimal (Single)')
    plt.plot(rewardSPMO1, timeTraceSPMO1, label='WSW-S2\n(AMT(<500))')
    plt.plot(rewardSPMO2, timeTraceSPMO2, label='OSW-S2\n(AMT(≥500))')
    plt.plot(rewardAWMO, timeTraceAWMO, label='AW-S2')
    plt.axvline(x = target[i], color='r', linestyle=':', label='#Required Number of \nHigh-quality Results')
    plt.xlabel('#High-quality Results')
    plt.ylabel('Time Utilization (Minute)')
    #plt.tight_layout()
    #plt.title('Time Utilization of Single Worker Set Method on Task 1')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    fname = './resultFiguresReal/timebyHighqualityResultsSingle' + day2 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()



    #------------------loss-----------------------
    maxMOIndex = np.argmax(gapCombiMO)
    maxSPMO1Index = np.argmax(gapCombiSPMO1)
    maxSPMO2Index = np.argmax(gapCombiSPMO2)
    maxAWMOIndex = np.argmax(gapCombiAWMO)

    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(rewardMO, np.log(gapCombiMO), marker='o', label='CrowdBwO-S2')
    plt.plot(rewardSPMO1, np.log(gapCombiSPMO1), marker='o', label='WSW-S2\n(AMT(<500))')
    plt.plot(rewardSPMO2, np.log(gapCombiSPMO2), marker='o', label='OSW-S2\n(AMT(≥500))')
    plt.plot(rewardAWMO, np.log(gapCombiAWMO), marker='o', label='AW-S2')
    
    plt.text(rewardMO[-1]+cordModify[i][0][0], np.log(gapCombiMO[-1])+cordModify[i][0][1], str((usedBudgetMO[-1]*para.omega + timeTraceMO[-1]*(1-para.omega)).round(2)) + '/' + str((optFeeTraceMO[-1]*para.omega + optTimeTraceMO[-1]*(1-para.omega)).round(2)), color='#66B2FF', fontweight='semibold', visible=visibleornot[i][0], fontsize=15)
    plt.text(rewardSPMO1[-1]+cordModify[i][1][0], np.log(gapCombiSPMO1[-1])+cordModify[i][1][1], str((usedBudgetSPMO1[-1]*para.omega + timeTraceSPMO1[-1]*(1-para.omega)).round(2)) + '/' + str((optFeeTraceSPMO1[-1]*para.omega + optTimeTraceSPMO1[-1]*(1-para.omega)).round(2)), color='#FF8000', fontweight='semibold', visible=visibleornot[i][1], fontsize=15)
    plt.text(rewardSPMO2[-1]+cordModify[i][2][0], np.log(gapCombiSPMO2[-1])+cordModify[i][2][1], str((usedBudgetSPMO2[-1]*para.omega + timeTraceSPMO2[-1]*(1-para.omega)).round(2)) + '/' + str((optFeeTraceSPMO2[-1]*para.omega + optTimeTraceSPMO2[-1]*(1-para.omega)).round(2)), color='#00CC00', fontweight='semibold', visible=visibleornot[i][2], fontsize=15)
    plt.text(rewardAWMO[-1]+cordModify[i][3][0], np.log(gapCombiAWMO[-1])+cordModify[i][3][1], str((usedBudgetAWMO[-1]*para.omega + timeTraceAWMO[-1]*(1-para.omega)).round(2)) + '/' + str((optFeeTraceAWMO[-1]*para.omega + optTimeTraceAWMO[-1]*(1-para.omega)).round(2)), color='#FF6666', fontweight='semibold', visible=visibleornot[i][3], fontsize=15)
    
    plt.text(rewardMO[maxMOIndex]+cordModify[i][4][0], np.log(gapCombiMO[maxMOIndex])+cordModify[i][4][1], str((usedBudgetMO[maxMOIndex]*para.omega + timeTraceMO[maxMOIndex]*(1-para.omega)).round(2)) + '/' + str((optFeeTraceMO[maxMOIndex]*para.omega + optTimeTraceMO[maxMOIndex]*(1-para.omega)).round(2)), color='#66B2FF', fontweight='semibold', visible=visibleornot[i][4], fontsize=15)
    plt.text(rewardSPMO1[maxSPMO1Index]+cordModify[i][5][0], np.log(gapCombiSPMO1[maxSPMO1Index])+cordModify[i][5][1], str((usedBudgetSPMO1[maxSPMO1Index]*para.omega + timeTraceSPMO1[maxSPMO1Index]*(1-para.omega)).round(2)) + '/' + str((optFeeTraceSPMO1[maxSPMO1Index]*para.omega + optTimeTraceSPMO1[maxSPMO1Index]*(1-para.omega)).round(2)), color='#FF8000', fontweight='semibold', visible=visibleornot[i][5], fontsize=15)
    plt.text(rewardSPMO2[maxSPMO2Index]+cordModify[i][6][0], np.log(gapCombiSPMO2[maxSPMO2Index])+cordModify[i][6][1], str((usedBudgetSPMO2[maxSPMO2Index]*para.omega + timeTraceSPMO2[maxSPMO2Index]*(1-para.omega)).round(2)) + '/' + str((optFeeTraceSPMO2[maxSPMO2Index]*para.omega + optTimeTraceSPMO2[maxSPMO2Index]*(1-para.omega)).round(2)), color='#00CC00', fontweight='semibold', visible=visibleornot[i][6], fontsize=15)
    plt.text(rewardAWMO[maxAWMOIndex]+cordModify[i][7][0], np.log(gapCombiAWMO[maxAWMOIndex])+cordModify[i][7][1], str((usedBudgetAWMO[maxAWMOIndex]*para.omega + timeTraceAWMO[maxAWMOIndex]*(1-para.omega)).round(2)) + '/' + str((optFeeTraceAWMO[maxAWMOIndex]*para.omega + optTimeTraceAWMO[maxAWMOIndex]*(1-para.omega)).round(2)), color='#FF6666', fontweight='semibold', visible=visibleornot[i][7], fontsize=15)
    

    plt.axvline(x = target[i], color='r', linestyle=':', label='#Required\nHigh-quality\nResults')
    plt.axhline(y = np.log(100), color='gray', linestyle=':', label='100%')
    plt.xlabel('#High-quality Results')
    plt.xticks(np.arange(0, ticks[i], 10))
    yticksLabels = ['$e^{' + str(j) + '}$' for j in np.arange(4, 10, 2)]
    plt.yticks(ticks=np.arange(4, 10, 2), labels=yticksLabels)
    plt.ylabel('Percentage of the\nOptimal\'s Combina-\ntorial Cost (%)')
    #plt.title('Combinatorial Cost Comparison on Task 1')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    #plt.tight_layout()
    fname = './resultFiguresReal/combiLoss' + day1 + day2 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()
    


    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(rewardMO, gapTimeMO, marker='o', label='CrowdBwO-S2')
    plt.plot(rewardSPMO1, gapTimeSPMO1, marker='o', label='WSW-S2\n(AMT(<500))')
    plt.plot(rewardSPMO2, gapTimeSPMO2, marker='o', label='OSW-S2\n(AMT(≥500))')
    plt.plot(rewardAWMO, gapTimeAWMO, marker='o', label='AW-S2')
    
    plt.text(rewardMO[-1]-1, gapTimeMO[-1]+1, str((timeTraceMO[-1]).round(2)) + '/' + str((optTimeTraceMO[-1]).round(2)), color='#66B2FF', fontweight='semibold', fontsize=15)
    plt.text(rewardSPMO1[-1]-1, gapTimeSPMO1[-1]+1, str((timeTraceSPMO1[-1]).round(2)) + '/' + str((optTimeTraceSPMO1[-1]).round(2)), color='#FF8000', fontweight='semibold', fontsize=15)
    plt.text(rewardSPMO2[-1]-1, gapTimeSPMO2[-1]+1, str((timeTraceSPMO2[-1]).round(2)) + '/' + str((optTimeTraceSPMO2[-1]).round(2)), color='#00CC00', fontweight='semibold', fontsize=15)
    plt.text(rewardAWMO[-1]-1, gapTimeAWMO[-1]+1, str((timeTraceAWMO[-1]).round(2)) + '/' + str((optTimeTraceAWMO[-1]).round(2)), color='#FF6666', fontweight='semibold', fontsize=15)
    
    plt.axvline(x = target[i], color='r', linestyle=':', label='#Required\nHigh-quality\nResults')
    plt.xlabel('#High-quality Results')
    plt.xticks(np.arange(0, ticks[i], 10))
    plt.ylabel('Percentage of the Optimal\'s \nTime Utilization (%)')
    
    #plt.title('Time Utilization Comparison on Task 1')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    #plt.tight_layout()
    fname = './resultFiguresReal/timeLoss' + day1 + day2 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()


    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.plot(rewardMO, gapFeeMO, marker='o', label='CrowdBwO-S2')
    plt.plot(rewardSPMO1, gapFeeSPMO1, marker='o', label='WSW-S2\n(AMT(<500))')
    plt.plot(rewardSPMO2, gapFeeSPMO2, marker='o', label='OSW-S2\n(AMT(≥500))')
    plt.plot(rewardAWMO, gapFeeAWMO, marker='o', label='AW-S2')
    
    plt.text(rewardMO[-1]-1, gapFeeMO[-1]+1, str((usedBudgetMO[-1]).round(2)) + '/' + str((optFeeTraceMO[-1]).round(2)), color='#66B2FF', fontweight='semibold', fontsize=15)
    plt.text(rewardSPMO1[-1]-1, gapFeeSPMO1[-1]+1, str((usedBudgetSPMO1[-1]).round(2)) + '/' + str((optFeeTraceSPMO1[-1]).round(2)), color='#FF8000', fontweight='semibold', fontsize=15)
    plt.text(rewardSPMO2[-1]-1, gapFeeSPMO2[-1]+1, str((usedBudgetSPMO2[-1]).round(2)) + '/' + str((optFeeTraceSPMO2[-1]).round(2)), color='#00CC00', fontweight='semibold', fontsize=15)
    plt.text(rewardAWMO[-1]-1, gapFeeAWMO[-1]+1, str((usedBudgetAWMO[-1]*para.omega + timeTraceAWMO[-1]).round(2)), color='#FF6666', fontweight='semibold', fontsize=15)
    
    plt.axvline(x = target[i], color='r', linestyle=':', label='#Required\nHigh-quality\nResults')
    plt.xlabel('#High-quality Results')
    plt.xticks(np.arange(0, ticks[i], 10))
    plt.ylabel('Percentage of the Optimal\'s \nExpense (%)')
    #plt.title('Expense Comparison on Task 1')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, framealpha=0.3)
    #plt.tight_layout()
    fname = './resultFiguresReal/expenseLoss' + day1 + day2 + '.png'
    plt.savefig(fname, bbox_inches=None, pad_inches=None)
    #plt.show()


    plt.close('all')

