import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Parameters import para


algs = ['CrowdBwO-S2', 'OWOP-S2']
width = 0.0027
barColors = ['#e07165', '#e0b965', '#a9e065', '#79c5e0', '#6588e0', '#c265e0']
barLabels = ['A-1', 'A-2', 'B-1', 'B-2', 'C-1', 'C-2']
#barLabels += [['_no_legend_'] * 6]


plt.rcParams.update({'font.size': 30})


for n in range(para.workerGroupsNumber):
    plt.figure(figsize=(10, 7))
    
    index = 0

    for alg in algs:
        plt.subplot(2, 1, index + 1)
        

        globals()[alg + 'df'] = pd.read_csv('./expAVEResults/' + alg + 'averageResultsWS' + str(n) + '.csv')

        globals()[alg + 'selectedArms'] = globals()[alg + 'df'][str(para.nArms[n] * 0)]
        globals()[alg + 'rewards'] = globals()[alg + 'df'][str(para.nArms[n] * 1)]
        globals()[alg + 'assignments'] = globals()[alg + 'df'][str(para.nArms[n] * 2)]

        if para.nArms[n] > 1:
            for arm in range(1, para.nArms[n]):
                globals()[alg + 'selectedArms'] = np.column_stack((globals()[alg + 'selectedArms'], globals()[alg + 'df'][str(arm * 1)]))
                globals()[alg + 'rewards'] = np.column_stack((globals()[alg + 'rewards'], globals()[alg + 'df'][str(arm + para.nArms[n] * 1)]))
                globals()[alg + 'assignments'] = np.column_stack((globals()[alg + 'assignments'], globals()[alg + 'df'][str(arm + para.nArms[n] * 2)]))
        globals()[alg + 'budget'] = np.array(globals()[alg + 'df'][str(3 * para.nArms[n])])
        globals()[alg + 'time'] = np.array(globals()[alg + 'df'][str(3 * para.nArms[n] + 1)])

        
        pulledTimePercentages = globals()[alg + 'selectedArms'] / np.tile(globals()[alg + 'time'].reshape(para.expNumber, 1), (1, para.nArms[n]))
        pulledTimePercentages = np.transpose(pulledTimePercentages).round(2)
        pulledTimePercentages = (pulledTimePercentages * 100)

        print(pulledTimePercentages, alg)
        

        for a in range(para.nArms[n]):
            bars = plt.bar((1 - para.omega[:para.expNumber])[::-1] - ((width * para.nArms[n]) / 2) + width * (a + 0.5), (pulledTimePercentages[a])[::-1], width, color=barColors[a], edgecolor='#5d5e5e', label=barLabels[a])

        index += 1

    

        '''if index == 1:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=23)
            #plt.ylabel('Task Publishing Time\nAllocation(%)', fontsize=20)'''

        plt.xticks((1 - para.omega[:para.expNumber])[::-1], fontsize=22)
        plt.yticks(np.arange(0, 101, 25), fontsize=22)
        plt.ylabel(alg, fontsize=25)


   
    plt.xlabel('Omega(500 High-quality Results Required)')
    plt.figtext(0.005, 0.16, 'Time Allocation(%)', fontsize=30, rotation='vertical')
    figurename = 'selectedRatesWS' + str(n+1)


    #plt.xticks(para.budget[:para.expNumber])

    '''plt.ylim(0, 300)
    plt.xlim(0, 300)'''


    plt.tight_layout()
    plt.savefig('./figuresAVEResults/S2' + figurename + '.png', bbox_inches=None, pad_inches=None)

    #plt.show()