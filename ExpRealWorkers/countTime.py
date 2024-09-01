import numpy as np
from AMTexperimentA.parameter import para


timeLimit = para.timeLimit
budget = para.budget



def timeCountS1(estimatedFeePara1, estimatedFeePara2):
    #record the traces of estimated results
    for t in range(timeLimit+1):
        if estimatedFeePara1 * t <= budget:
            t1 = t
        if estimatedFeePara2 * t <= budget:
            t2 = t
        if (estimatedFeePara1 + estimatedFeePara2) * t <= budget:
            t3 = t

    return t1, t2, t3




def timeCountS2(estimatedAccuracy1, estimatedAccuracy2, estimatedTrafficPara1, estimatedTrafficPara2, requiredResults):
    #record the traces of estimated results
    a = np.zeros(3)
    t = 0
    while np.any(a == 0) == 1:
        t += 1
        if estimatedAccuracy1 * estimatedTrafficPara1 * t >= requiredResults:
            if a[0] == 0:
                t1 = t
                a[0] = 1
        if estimatedAccuracy2 * estimatedTrafficPara2 * t >= requiredResults:
            if a[1] == 0:
                t2 = t
                a[1] = 1
        if (estimatedAccuracy1 * estimatedTrafficPara1 + estimatedAccuracy2 * estimatedTrafficPara2) * t >= requiredResults:  
            if a[2] == 0:
                t3 = t
                a[2] = 1


    return t1, t2, t3