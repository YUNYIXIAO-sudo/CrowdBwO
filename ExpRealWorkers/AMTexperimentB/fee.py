import numpy as np

feePara = np.array([3, 3]) #the incentive is $0.02, fee to platform is $0.01

'''20% fee on the reward and bonus amount (if any) you pay Workers. 
Tasks with 10 or more assignments will be charged an additional 20% fee 
on the reward you pay Workers. 
The minimum fee is $0.01 per assignment or bonus payment. --amt'''

def feeCount(Workers):
    fee = Workers * feePara

    fee[Workers == 0] = 0
    
    return np.sum(fee)

