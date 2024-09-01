import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Parameters import para


algs = ['CrowdBwO-S2', 'AW-S2', 'BwK-GD', 'BB-S2', 'OWOP-S2', 'OSWOP-S2', 'OSW-S2', 'WSW-S2']
lineColor = ['#FDAC53', '#9BB7D4', '#B55A30', '#E9897E', '#A0DAA9', '#0072B5', '#926AA6', '#E0B589']

plt.rcParams.update({'font.size': 22})


for n in range(para.workerGroupsNumber):

    for i in range(3):
        plt.figure(figsize=(12, 6))
        failedAlg = {}
      

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


            #print(n, globals()[algs[a] + 'budget'], globals()[algs[a] + 'time'], alg)


            if np.all(np.sum(globals()[algs[a] + 'rewards'], axis=1) < para.answersRequired[0]):
                failedAlg[algs[a]] = np.max(np.sum(globals()[algs[a] + 'rewards'], axis=1)).astype(np.int64)
                

            else:
                plotIndex = np.where(np.sum(globals()[algs[a] + 'rewards'], axis=1) >= para.answersRequired[0])[0]

                if i == 0:
                    plt.plot((1 - para.omega[plotIndex])[::-1], (para.omega[plotIndex] * globals()[algs[a] + 'budget'][plotIndex] + (1 - para.omega[plotIndex]) * globals()[algs[a] + 'time'][plotIndex])[::-1], marker='o', color=lineColor[a], label=algs[a], linewidth=5, markeredgewidth=7)
                elif i == 1:
                    plt.plot((1 - para.omega[plotIndex])[::-1], (globals()[algs[a] + 'time'][plotIndex])[::-1], marker='o', color=lineColor[a], label=algs[a], linewidth=5, markeredgewidth=7)
                elif i == 2:
                    plt.plot((1 - para.omega[plotIndex])[::-1], (globals()[algs[a] + 'budget'][plotIndex])[::-1], marker='o', color=lineColor[a], label=algs[a], linewidth=5, markeredgewidth=7)
                    

        
        
        
        failedAlg = sorted(failedAlg.items(), key=lambda item: item[1], reverse=True)
        failedAlg = dict(failedAlg)
        failed = 'Failed: \n'
        for key, item in failedAlg.items():
            failed += key + '(' + str(item) + ') '
        failed = str(failed)[:len(failed)-1]
        #failed = failed[:41] + '\n' + failed[41:]
        
        

        plt.title(failed, fontsize=20, color='r', ha='left', position=(-0.1, 1))
                
        if i == 0:
            plt.ylabel('Combinatorial Cost')
            figurename = 'combicostbyomegaWS' + str(n+1)
        elif i == 1:
            plt.ylabel('Time Cost')
            figurename = 'timecostbyomegaWS' + str(n+1)
        elif i == 2:
            plt.ylabel('Expense Cost')
            figurename = 'expensecostbyomegaWS' + str(n+1)


        plt.xlabel('omega (500 High-quality Results are Required)')
        plt.tight_layout()
        plt.xticks((1 - para.omega[:para.expNumber])[::-1])
        

        '''plt.ylim(0, 300)
        plt.xlim(0, 300)'''

        plt.legend(fontsize=15)
        plt.savefig('./figuresAVEResults/S2' + figurename + '.png', bbox_inches=None, pad_inches=None)

        #plt.show()