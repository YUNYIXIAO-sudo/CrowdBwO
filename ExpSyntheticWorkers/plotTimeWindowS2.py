import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Parameters import para
from NewTrafficFunction import estimateTraffic

armName = ['A-1(3)', 'A-2(12)', 'B-1(3)', 'B-2(12)', 'C-1(3)', 'C-2(3)']
lineColors = ['#e07165', '#e0b965', '#a9e065', '#79c5e0', '#6588e0', '#c265e0']
mtime = np.array([3, 12, 3, 12, 3, 3])



plt.rcParams.update({'font.size': 30})



for n in range(para.workerGroupsNumber):
    theta0 = np.array(para.theta0[n], dtype=np.float64)
    theta1 = np.array(para.theta1[n], dtype=np.float64)
    theta2 = np.array(para.theta2[n], dtype=np.float64)
    theta3 = np.array(para.theta3[n], dtype=np.float64)

    thetas = np.vstack((theta0, theta1, theta2, theta3))
    estimatedTraffic, estimatedFee, mtime = estimateTraffic.getTrafficNoTimeLimit(thetas, np.zeros(shape=(4, para.nArms[n])), 0, n)

    df = pd.read_csv('./expAVEResults/CrowdBwO-S2averageResultsWS' + str(n) + '.csv')

    selectedArms = df[str(para.nArms[n] * 0)]
    
    if para.nArms[n] > 1:
        for arm in range(1, para.nArms[n]):
            selectedArms = np.column_stack((selectedArms, df[str(arm * 1)]))
    time = np.array(df[str(3 * para.nArms[n] + 1)])
    

    
    pulledTimePercentages = selectedArms / np.tile(time.reshape(para.expNumber, 1), (1, para.nArms[n]))
    pulledTimePercentages = np.transpose(pulledTimePercentages).round(2)
    pulledTimePercentages = (pulledTimePercentages * 100).astype(int)

    plt.figure(figsize=(10, 5))

    

    for a in range(para.nArms[n]):
        df = pd.read_csv('./expAVETimeModelResults/CrowdBwO-S2TimeModelAverageResultsWS' + str(n) + '.csv')

        '''thetas0 = df[str(a)]
        thetas1 = df[str(para.nArms[n] + a)]
        thetas2 = df[str(para.nArms[n] * 2 + a)]
        thetas3 = df[str(para.nArms[n] * 3 + a)]'''
        cutTime = np.array(df[str(para.nArms[n] * 8 + a)])
        #countCutTime = np.array(df[str(para.nArms[n] * 9 + a)])

        plt.plot((1 - para.omega[:para.expNumber])[::-1], cutTime[::-1], color=lineColors[a], label=armName[a], linewidth=5)
        plt.scatter((1 - para.omega[:para.expNumber])[::-1], cutTime[::-1], s=(pulledTimePercentages[a]*6)[::-1], color=lineColors[a])
 
        
    plt.axhline(y=3, color='grey', linestyle=':', linewidth=4)
    plt.axhline(y=12, color='grey', linestyle=':', linewidth=4)

    
    plt.xticks((1 - para.omega[:para.expNumber])[::-1], fontsize=22)
    plt.yticks(np.arange(0, 19, 5), fontsize=22)
    plt.xlabel('omega(500 High-quality Results Required)', fontsize=27)
    plt.ylabel('Average Publish\nWindow Length of\neach Worker Set')
    figurename = 'CrowdBwO-S2windowLengthWS' + str(n+1)
    
    #plt.xticks(para.budget[:para.expNumber])

    '''plt.ylim(0, 300)
    plt.xlim(0, 300)'''

    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=25)
    plt.tight_layout()
    plt.savefig('./figuresAVEResults/S2' + figurename + '.png', bbox_inches=None, pad_inches=None)

    #plt.show()