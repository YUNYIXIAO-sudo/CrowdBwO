import numpy as np


class para():
   expNumber = 10
   workerGroupsNumber = 2
   
   
   #worker groups information
   rewardProbs = np.array([[0.3, 0.6, 0.3, 0.6], 
                           [0.3, 0.6, 0.3, 0.6, 0.4, 0.5]
                           ], dtype=object)
   theta0 = np.array([[0.05, 0.05, 0.05, 0.05], 
                     [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
                  ], dtype=object) #real thetas for worker traffic
   theta1 = np.array([[0.5, 0.5, 0.5, 0.5], 
                     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                  ], dtype=object)
   theta2 = np.array([[65, 50, 65, 50], 
                     [65, 50, 65, 50, 55, 55]
                  ], dtype=object)
   theta3 = np.array([[120, 80, 120, 80], 
                     [120, 80, 120, 80, 110, 100]
                  ], dtype=object)
   feePara = np.array([[6, 6, 3, 3],
                     [6, 6, 3, 3, 4, 4]], dtype=object) #real fee counting paras
   minimumFee = np.array([[18, 18, 6, 6],
                        [18, 18, 6, 6, 8, 8]], dtype=object) 
   batchCreateTime = np.array([[2, 2, 2, 2],
                              [2, 2, 2, 2, 2, 2]], dtype=object) #time for creating a batch for each project
   

   nArms = np.array([4, 6])


   #environment information
   timeLimit = np.ones(expNumber).astype(np.int64) * 500
   budget = np.arange(1, expNumber+1, 1) * 1000
   answersRequired = np.ones(expNumber) * 500
   omega = np.arange(1, expNumber+1, 1) * 0.02
  


   #parameters
   totalRoundsBBS2 = np.ones(shape=(workerGroupsNumber, expNumber)).astype(np.int64) * 750
   
   totalRoundsBwKS1 = np.array([[21, 42, 63, 84, 104, 111, 111, 111, 111, 111],
                              [16, 32, 47, 63, 78, 94, 105, 105, 105, 105]])
   
   
   
   '''
   totalRoundsBB = np.array([[3900, 12800, 6000, 23000, 23000, 23000],
                            [4100, 13600, 6000, 23000, 23000, 23000]])
   '''

   #trainWindow = np.ones(expNumber).astype(int) * 60
   S2timeLimit = np.ones(expNumber).astype(np.int64) * 720
   zeta = np.ones(expNumber) * 1
   n_iter = np.ones(expNumber).astype(np.int64) * 50
   delta = np.ones(expNumber) * 0.1
   beta = np.ones(expNumber) * 0.2
   publishWindow = np.ones(expNumber) * 10
    



