import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Parameters import para


algs = ['CrowdBwO-S1', 'AW-S1', 'BwK-RRS', 'BB-S1', 'OWOP-S1', 'OSWOP-S1', 'OSW-S1', 'WSW-S1']
lineColor = ['#FDAC53', '#9BB7D4', '#B55A30', '#E9897E', '#A0DAA9', '#0072B5', '#926AA6', '#E0B589']

plt.rcParams.update({'font.size': 22})

for n in range(para.workerGroupsNumber):

    plt.figure(figsize=(11, 8))


    for a in range(len(algs)):
        globals()[algs[a] + 'df'] = pd.read_csv('./expAVEResults/' + algs[a] + 'averageResultsWS' + str(n) + '.csv')

        globals()[algs[a] + 'selectedArms'] = globals()[algs[a] + 'df'][str(para.nArms[n] * 0)]
        globals()[algs[a] + 'rewards'] = globals()[algs[a] + 'df'][str(para.nArms[n] * 1)]
        globals()[algs[a] + 'assignments'] = globals()[algs[a] + 'df'][str(para.nArms[n] * 2)]

        if para.nArms[n] > 1:
            for arm in range(1, para.nArms[n]):
                globals()[algs[a] + 'selectedArms'] = np.column_stack((globals()[algs[a] + 'selectedArms'], globals()[algs[a] + 'df'][str(arm * 1)]))
                globals()[algs[a] + 'rewards'] = np.column_stack((globals()[algs[a] + 'rewards'], globals()[algs[a] + 'df'][str(arm + para.nArms[n] * 1)]))
                globals()[algs[a] + 'assignments'] = np.column_stack((globals()[algs[a] + 'assignments'], globals()[algs[a] + 'df'][str(arm + para.nArms[n] * 2)]))
        globals()[algs[a] + 'budget'] = np.array(globals()[algs[a] + 'df'][str(3 * para.nArms[n])])
        globals()[algs[a] + 'time'] = np.array(globals()[algs[a] + 'df'][str(3 * para.nArms[n] + 1)])

      

        plt.plot(para.budget[:para.expNumber], np.sum(globals()[algs[a] + 'rewards'], axis=1), marker='o', color=lineColor[a], label=algs[a], linewidth=5, markeredgewidth=7)


    plt.xticks(para.budget[:para.expNumber])
    plt.xlabel('Expense Budget (Time Budget = 500)')
    plt.ylabel('Average #High-quality Results')
    figurename = 'rewardsbyepxensebudgetWS' + str(n+1)

    plt.tight_layout()
    #plt.xticks(para.budget[:para.expNumber])

    '''plt.ylim(0, 300)
    plt.xlim(0, 300)'''

    plt.legend(fontsize=19)
    plt.savefig('./figuresAVEResults/S1' + figurename + '.png', bbox_inches=None, pad_inches=None)

    #plt.show()