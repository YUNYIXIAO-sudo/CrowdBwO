import numpy as np
from WorkerTraffic import workerTraffic
#from DesignBatch import designBatch
from Parameters import para
from environment import env, feeCount, feeCountSeperate
import pandas as pd
from TB_MOUCB import TB_MOUCBalgorithm
from NewTrafficFunction import estimateTraffic
from DesignBatchNonTimeModeling import designBatch
import time

def logistic(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(32)
updateTimeModelRate = 4

class simulation(object):
    def __init__(self):
        self.pulledTimeR = [para.nArms[workerset] * [0]]
        self.rewardsR = np.zeros(para.nArms[workerset])
        self.pulledCountsR = np.zeros(para.nArms[workerset])
        self.BudgetR = [[0]]
        self.tR = [[0]]
        self.thetas0R = np.zeros(para.nArms[workerset])
        self.thetas_std0R = np.zeros(para.nArms[workerset])
        self.thetas1R = np.zeros(para.nArms[workerset])
        self.thetas_std1R = np.zeros(para.nArms[workerset])
        self.thetas2R = np.zeros(para.nArms[workerset])
        self.thetas_std2R = np.zeros(para.nArms[workerset])
        self.thetas3R = np.zeros(para.nArms[workerset])
        self.thetas_std3R = np.zeros(para.nArms[workerset])

        self.cutTimeR = np.zeros(para.nArms[workerset])
        self.countCutTimeR = np.zeros(para.nArms[workerset])


    def firstTrial(self):
        beta = para.beta[experiment]

        NumberofAnswer=para.answersRequired[experiment]
        answerMax = NumberofAnswer * beta

        firstRealWorkers = np.zeros(para.nArms[workerset])
        firstRewards = np.zeros(para.nArms[workerset])
        fee = 0

        while np.sum(firstRewards) < answerMax:

            if self.t >= np.array(para.batchCreateTime[workerset], dtype=np.int32)[0]:
            
                '''pulledCounts = 
                [[x11, x12, ..., x1k], 
                [x21, x22, ..., x2k], 
                ... , 
                [xbatchTime1, xbatchTime2, ..., xbatchTimek]]'''
                
                firstRealWorkers = env.reactWorkerNum(self.pullingArms, self.t, np.sum(self.pulledCounts, axis=0), experiment, workerset)
                
                firstRewards = env.reactRewards(self.pullingArms, firstRealWorkers, experiment, workerset)

                timeStamp = self.t - np.array(para.batchCreateTime[workerset], dtype=np.int32)

                self.timeStamp = np.vstack((self.timeStamp, timeStamp))
                self.pulledRec = np.vstack((self.pulledRec, firstRealWorkers))

                fee = feeCount(firstRealWorkers, experiment, workerset)
                
            self.t += 1
            self.pulledTime += 1
        
        
        self.pulledCounts = firstRealWorkers.copy()
        self.rewards = firstRewards.copy()
        self.usedBudget = fee

        #delete the first 0 data in self.timeStamp and self.pulledRec
        self.timeStamp = np.delete(self.timeStamp, 0, axis=0)
        self.pulledRec = np.delete(self.pulledRec, 0, axis=0)


        print('first', self.usedBudget, self.t, self.pulledCounts)


    def sim(self, Agent):
        NumberofAnswer=para.answersRequired[experiment]
        N = para.n_iter[experiment]
        countModelTrain = 0
        countTrial = 0

        for n in range(N): #number of simulations
            #Initializations
            agent = Agent(experiment, workerset)
            pullingIsFeasible = 1

            self.pullingArms = np.zeros(para.nArms[workerset])
            self.rewards = np.zeros(para.nArms[workerset])
            self.pulledCounts = np.zeros(para.nArms[workerset])
            self.pulledRec = np.zeros(para.nArms[workerset])
            self.timeStamp = np.zeros(para.nArms[workerset])
            self.thetas = np.ones(3)
            self.thetas_std = np.ones(3)
            self.pulledTime = np.zeros(para.nArms[workerset])
            self.cutTime = np.zeros(para.nArms[workerset])
            self.countCutTime = np.zeros(para.nArms[workerset])


            self.t = 0
            self.m = 0
            self.tau = 0
            self.usedBudget = 0
            plusTime = np.zeros(para.nArms[workerset])
            Avar = 1
            

            '''self.pulledTimeR.append([])
            self.rewardsR.append([])
            self.pulledCountsR.append([])
            self.BudgetR.append([])
            self.tR.append([])'''

            #Initializations end

            #round loops
            while pullingIsFeasible == 1:

                if self.m == 0:
                    self.pullingArms = np.ones(para.nArms[workerset])

                    self.firstTrial()
                    realWorkers = np.zeros(para.nArms[workerset])
                    rewardThisTime = np.zeros(para.nArms[workerset])
                    addWorkersforRecords = np.zeros(para.nArms[workerset])
                    addRewardsforRecords = np.zeros(para.nArms[workerset])
                    addFeeforRecords = 0


                    self.m += 1
                    countTrial += 1
                    self.tau = self.t
                

                elif self.m > 0:
                    if self.t == self.tau:
                        updateFrequency = np.floor(updateTimeModelRate / Avar)

                        if self.m % updateFrequency == 1:
                            self.thetas, self.thetas_std = workerTraffic.train(self.timeStamp, self.pulledRec, experiment, workerset)
                            countModelTrain += 1
                        Avar = logistic(np.mean(self.thetas_std))
                      
                        



                        agent.sample(self.pulledCounts + realWorkers, self.rewards + rewardThisTime)

                        estimatedTraffic, estimatedFee, cutTime = estimateTraffic.getTrafficNoTimeLimit(self.thetas, self.thetas_std, experiment, workerset)
                        '''if workerset == 1:
                            print('m:', self.m, 'cutTime:', cutTime, 'timemodel:', self.thetas)'''
                        
                        estimatedDensityTraffic, estimatedDensityFee, timeDensity = estimateTraffic.getDensityTrafficNoTimeLimit(self.thetas, self.thetas_std, experiment, workerset)


                        pullingArms, cuttingMethod = agent.get_arms(NumberofAnswer - np.sum(self.rewards), estimatedTraffic, estimatedFee, estimatedDensityTraffic, estimatedDensityFee)
                        
                        

                        pullingArmsThisTime = np.array([np.random.binomial(1, p) for p in pullingArms])
                        #print('Check Arms:', pullingArms, pullingArmsThisTime, self.pulledTime, estimatedTraffic, estimatedFee)
                        
                
                        sameArms = np.where(self.pullingArms == pullingArmsThisTime)[0]


                        self.pullingArms = pullingArmsThisTime


                        self.m += 1
                        countTrial += 1
                        self.batchTime = np.zeros(para.nArms[workerset])
                        self.batchTime[sameArms] += plusTime[sameArms]


                        batchTime = designBatch.arithmeticNoTimeLimit(cutTime, pullingArmsThisTime)
                        self.tau += batchTime
                        #print(self.rewards, self.pulledRec, self.m, self.t, self.pullingArms, self.usedBudget, time, 'result of a batch')
                       

                    self.batchTime += 1


                    timeStamp = self.batchTime % cutTime 
                    timeStamp[timeStamp == 0] = cutTime[timeStamp == 0]
                    timeStamp = timeStamp * pullingArmsThisTime
                   
                    plusTime = timeStamp.copy()

                    if np.any(timeStamp > np.array(para.batchCreateTime[workerset], dtype=np.int32)) == 1:
                        realWorkers = env.reactWorkerNum(pullingArmsThisTime, np.array(timeStamp), np.sum(self.pulledCounts, axis=0), experiment, workerset)
                        #create time -> worker = 0
                        
                        self.pulledCounts[timeStamp == cutTime] += realWorkers[timeStamp == cutTime]
                        addWorkersforRecords = np.where(timeStamp == cutTime, 0, realWorkers)

                        rewardThisTime = env.reactRewards(pullingArmsThisTime, realWorkers, experiment, workerset)
                        self.rewards[timeStamp == cutTime] += rewardThisTime[timeStamp == cutTime]
                        addRewardsforRecords = np.where(timeStamp == cutTime, 0, rewardThisTime)

                        fee = feeCountSeperate(realWorkers, experiment, workerset)
                        feetoCount = np.where(timeStamp != cutTime, 0, fee)
                        self.usedBudget += np.sum(feetoCount)
                        addFeeforRecords = np.sum(np.where(timeStamp == cutTime, 0, fee))

                      
                        #reform timestamp and worker number data for model training
                        timeStamp -= np.array(para.batchCreateTime[workerset], dtype=np.int32)
                    
                        timeStampNoZero = np.where(timeStamp < 0, 0, timeStamp)

                        #print(realWorkers, timeStamp, self.pulledRec[0][0], 'check')

                        self.timeStamp = np.vstack((self.timeStamp, timeStampNoZero))
                        self.pulledRec = np.vstack((self.pulledRec, realWorkers))

                        timeStamp += np.array(para.batchCreateTime[workerset], dtype=np.int32)

                        realWorkers[timeStamp == cutTime] = 0
                        rewardThisTime[timeStamp == cutTime] = 0

                    
                    self.t += 1
                    self.pulledTime += self.pullingArms
                    self.cutTime[timeStamp == cutTime] += timeStamp[timeStamp == cutTime]
                    self.countCutTime[timeStamp == cutTime] += 1
                    '''if workerset == 1:
                        print('m:', self.m, 't:', self.t, 'self.cutTime:', self.cutTime, 'countCutTime:', self.countCutTime)'''


                    if np.sum(self.rewards) >= NumberofAnswer or self.t >= para.S2timeLimit[experiment]:
                        leftcutTimeIndex = np.where((timeStamp > 0) & (timeStamp < cutTime))[0]
                       
                        self.cutTime[leftcutTimeIndex] += cutTime[leftcutTimeIndex]
                        self.countCutTime[leftcutTimeIndex] += 1
                      
                        pullingIsFeasible = 0

                        self.recordsPrint(addWorkersforRecords, addRewardsforRecords, addFeeforRecords)
        

        self.pulledTimeR = np.delete(self.pulledTimeR, 0, 0)
        self.rewardsR = np.delete(self.rewardsR, 0, 0)
        self.pulledCountsR = np.delete(self.pulledCountsR, 0, 0)
        self.BudgetR = np.delete(self.BudgetR, 0, 0)
        self.tR = np.delete(self.tR, 0, 0)
        self.thetas0R = np.delete(self.thetas0R, 0, 0)
        self.thetas_std0R = np.delete(self.thetas_std0R, 0, 0)
        self.thetas1R = np.delete(self.thetas1R, 0, 0)
        self.thetas_std1R = np.delete(self.thetas_std1R, 0, 0)
        self.thetas2R = np.delete(self.thetas2R, 0, 0)
        self.thetas_std2R = np.delete(self.thetas_std2R, 0, 0)
        self.thetas3R = np.delete(self.thetas3R, 0, 0)
        self.thetas_std3R = np.delete(self.thetas_std3R, 0, 0)

        self.cutTimeR = np.delete(self.cutTimeR, 0, 0)
        self.countCutTimeR = np.delete(self.countCutTimeR, 0, 0)



        return self.pulledTimeR, self.rewardsR, self.pulledCountsR, self.BudgetR, self.tR, self.thetas0R, self.thetas_std0R, self.thetas1R, self.thetas_std1R, self.thetas2R, self.thetas_std2R, self.thetas3R, self.thetas_std3R, self.cutTimeR, self.countCutTimeR, countTrial, countModelTrain
            
    def recordsPrint(self, realWorkers, rewardThisTime, fee):
        self.pulledTimeR.append(self.pulledTime)
        self.rewardsR = np.vstack((self.rewardsR, self.rewards + rewardThisTime))
        self.pulledCountsR = np.vstack((self.pulledCountsR, self.pulledCounts + realWorkers))
        self.BudgetR.append([self.usedBudget + fee])
        self.tR.append([self.t])
        self.thetas0R = np.vstack((self.thetas0R, self.thetas[0]))
        self.thetas_std0R = np.vstack((self.thetas_std0R, self.thetas_std[0]))
        self.thetas1R = np.vstack((self.thetas1R, self.thetas[1]))
        self.thetas_std1R = np.vstack((self.thetas_std1R, self.thetas_std[1]))
        self.thetas2R = np.vstack((self.thetas2R, self.thetas[2]))
        self.thetas_std2R = np.vstack((self.thetas_std2R, self.thetas_std[2]))
        self.thetas3R = np.vstack((self.thetas3R, self.thetas[3]))
        self.thetas_std3R = np.vstack((self.thetas_std3R, self.thetas_std[3]))
        self.countCutTimeR = np.vstack((self.countCutTimeR, self.countCutTime))
        #avoid zero dividing
        self.countCutTime = np.where(self.cutTime==0, 1, self.countCutTime)
        self.cutTimeR = np.vstack((self.cutTimeR, self.cutTime / self.countCutTime))


f = open('./exeTimeMO.csv', mode='a')

for i in range(para.workerGroupsNumber):    
    workerset = i

    
    f = open('./expAVEResults/CrowdBwO-S2averageResultsWS' + str(i) + '.csv', mode='w')
    f2 = open('./expAVETimeModelResults/CrowdBwO-S2TimeModelAverageResultsWS' + str(i) + '.csv', mode='w')
    
    index = ''
    for a in range(para.nArms[i]*3+2):
        index += str(a)
        index += ','
    index = index[:-1]
    index += '\n'
    
    f.write(index)
    
    index = ''
    for a in range(para.nArms[i]*10):
        index += str(a)
        index += ','
    index = index[:-1]
    index += '\n'

    f2.write(index)
    f.close()
    f2.close()
    

    for j in range(para.expNumber):
        experiment = j

        if __name__ == "__main__":
            timeStart = time.time()

            si = simulation()

            selectedArmsR, rewardsR, pulledR, budgetR, tR, thetas0R, thetas_std0R, thetas1R, thetas_std1R, thetas2R, thetas_std2R, thetas3R, thetas_std3R, cutTimeR, countCutTimeR, m, modelTrain = si.sim(TB_MOUCBalgorithm)
            
            expN = len(tR) * [[experiment]]
            
            results = np.hstack((np.array(selectedArmsR), rewardsR, pulledR, np.array(budgetR), np.array(tR), expN))
            df = pd.DataFrame(results)
            #, columns=['Selected Project', 'Cumulative Rewards', 'Number of Assignments', 'Remaining Budget', 'Spent Time'])
            
            df.to_csv('./expResults/CrowdBwO-S2resultsW' + str(workerset) + 'E' + str(experiment) + '.csv', mode='w')

            averageResults = np.hstack((np.mean(np.array(selectedArmsR), axis=0), np.mean(rewardsR, axis=0), np.mean(pulledR, axis=0), np.mean(np.array(budgetR), axis=0), np.mean(np.array(tR), axis=0)))
            df = pd.DataFrame([averageResults])
            
            df.to_csv('./expAVEResults/CrowdBwO-S2averageResultsWS' + str(i) + '.csv', mode='a', index=False, header=False)

            timeModelAverageResults = np.hstack((np.mean(np.array(thetas0R), axis=0), np.mean(np.array(thetas1R), axis=0), np.mean(np.array(thetas2R), axis=0), np.mean(np.array(thetas3R), axis=0), np.mean(np.array(thetas_std0R), axis=0), np.mean(np.array(thetas_std1R), axis=0), np.mean(np.array(thetas_std2R), axis=0), np.mean(np.array(thetas_std3R), axis=0), np.mean(np.array(cutTimeR), axis=0), np.mean(np.array(countCutTimeR), axis=0)))
            df = pd.DataFrame([timeModelAverageResults])
            
            df.to_csv('./expAVETimeModelResults/CrowdBwO-S2TimeModelAverageResultsWS' + str(i) + '.csv', mode='a', index=False, header=False)
            

            timeFinish = time.time()
            timeExe = timeFinish - timeStart
            print('Time', timeExe, 'workergroup:', i, 'exp:', j, 'trials:', m)
            
            
            f.write('Time,' + str(timeExe / para.n_iter[experiment]) + ',workergroup,' + str(i) + ',exp,' + str(j) + ',trials,' + str(m / para.n_iter[experiment]) + ',modelTrain,' + str(modelTrain / para.n_iter[experiment]) + '\n')


f.close()