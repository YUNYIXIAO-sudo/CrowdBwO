from fee import feeCount
from DesignBatchNonTimeModeling import designBatch
from NewTrafficFunction import estimateTraffic
from WorkerTraffic import workerTraffic
import numpy as np
import time
import pandas as pd
from TB_UCB import TB_UCBalgorithm
from matplotlib import pyplot as plt
from parameter import para
from datetime import datetime


from revokeQaulification import revokeWorkers1, revokeWorkers2
from hitApproval1 import publishHIT1
from hitApproval2 import publishHIT2
from terminateHIT import stopHIT
from answers import analyzeResults 



f = open('ThroughputandFeeParaLP' + str(datetime.now().day) + '.csv', 'a')


def logistic(x):
    return 1 / (1 + np.exp(-x))

'''
def revokeWorkers1(id, qualification):
    pass

def revokeWorkers2(id, qualification):
    pass

def publishHIT1(qua):
    return 1

def publishHIT2(qua):
    return 2

def stopHIT(id):
    ids = id

def analyzeResults(workers, rewards, time, qualification):
    collectedAnswers = workers + np.sqrt(np.array([10, 70]) * time).astype(np.int64)
    correctAnswers = rewards + np.sqrt(np.array([6, 60]) * time).astype(np.int64)

    return collectedAnswers, correctAnswers'''
#remember to delete the time !!!!!!!



np.random.seed(32)

#mturk '38TZ8V8N0D707RXSEURUIX1GNICPIB'
qualification = '35K484E78ZRG654Y9Z9GHCOGH6DYYK'


class assigningProcess(object):
    def __init__(self):
        self.selectedArmsR = [[0, 0]]
        self.rewardsR = np.zeros(para.nArms)
        self.pulledCountsR = np.zeros(para.nArms)
        self.BudgetR = [[0]]
        self.tR = [[0]]
        self.meanR = [['0']]
        self.stdR = [['0']]
        self.updateFrequencyR = [[0]]
        self.realTimeR = [['0']]
        self.pulledTimeR = np.zeros(para.nArms)


    def firstTrial(self):
        task1Available = 0
        task2Available = 0
        B = para.budget
        T = para.timeLimit
        
        timeMax = T * para.beta
        
        publishTime = np.zeros(para.nArms)

        fee = 0
        realWorkers = np.zeros(para.nArms)
        rewardThisTime = np.zeros(para.nArms)

       
        hitid1 = publishHIT1(qualification)
        hitid2 = publishHIT2(qualification)
        if task1Available == 0:
            task1Available = 1
        else:
            print('1ERROR!!!!!!!!!!!!')

        if task2Available == 0:
            task2Available = 1
        else:
            print('2ERROR!!!!!!!!!!!!')


        while fee < (B * para.beta) and self.t < timeMax:
            time.sleep(60)
            self.pulledTime += self.pullingArms

            publishTime += 1
            self.t += 1

            '''pulledCounts = 
            [[x11, x12, ..., x1k], 
            [x21, x22, ..., x2k], 
            ... , 
            [xbatchTime1, xbatchTime2, ..., xbatchTimek]]'''

            self.realTime = str(datetime.now().strftime("%H:%M:%S"))
            if self.t > para.batchCreateTime:
                
                realWorkers, rewardThisTime = analyzeResults(qualification)
                self.pulledCounts = realWorkers.copy()
                self.rewards = rewardThisTime.copy()

                fee = feeCount(realWorkers)
                self.usedBudget = fee

                workerDiff = realWorkers.copy()
            
                timeStamp = publishTime - para.batchCreateTime
                

                self.timeStamp = np.vstack((self.timeStamp, timeStamp))
                self.pulledRec = np.vstack((self.pulledRec, workerDiff))

            print('FIRST    t: ' + str(self.t) + ' assign: ' + str(np.sum(realWorkers)) + ' fee: ' + str(np.sum(realWorkers) * 3 / 100))
            self.recordsPrint()
    
        stopHIT(hitid1)
        stopHIT(hitid2)
        revokeWorkers1(hitid1, qualification)
        revokeWorkers2(hitid2, qualification)


        if task1Available == 1:
            task1Available = 0
        else:
            print('3ERROR!!!!!!!!!!!!')

        if task2Available == 1:
            task2Available = 0
        else:
            print('4ERROR!!!!!!!!!!!!')
        

        self.timeStamp = np.delete(self.timeStamp, 0, axis=0)
        self.pulledRec = np.delete(self.pulledRec, 0, axis=0)

        #self.recordsPrint(self.rewards, np.sum(self.pulledCounts, axis=0), self.pullingArms, self.t, self.remainBudget)

        print('FIRST Final    t: ' + str(self.t) + ' assign: ' + str(np.sum(self.pulledCounts)) + ' fee: ' + str(self.usedBudget / 100))

        return hitid1, hitid2, realWorkers, rewardThisTime

    def assign(self, Agent):
        task1Available = 0
        task2Available = 0

        B = para.budget
        T = para.timeLimit

        #Initializations
        agent = Agent()
        pullingIsFeasible = 1

        self.pullingArms = np.zeros(para.nArms)
        self.rewards = np.zeros(para.nArms)
        self.pulledCounts = np.zeros(para.nArms)
        self.pulledRec = np.zeros(para.nArms)
        self.timeStamp = np.zeros(para.nArms)
        self.thetas = np.ones(3)
        self.thetas_std = np.ones(3)
        self.batchTime = np.zeros(para.nArms)
        self.pulledTime = np.zeros(para.nArms)
        self.updateFrequency = para.updateRate
        


        self.t = 0
        self.m = 0
        self.tau = 0
        self.usedBudget = 0

      
        Avar = 1
        realWorkers = np.zeros(para.nArms)
        rewardThisTime = np.zeros(para.nArms)


        '''self.selectedArmsR.append([])
        self.rewardsR.append([])
        self.pulledCountsR.append([])
        self.BudgetR.append([])
        self.tR.append([])'''

        #Initializations end

        #round loops
        while pullingIsFeasible == 1:

            if self.m == 0:
                self.pullingArms = np.ones(para.nArms)
                pullingArmsThisTime = np.ones(para.nArms)
                
                hitid1, hitid2, realWorkers, rewardThisTime = self.firstTrial()
                workerBefore = realWorkers.copy()
                

                self.m += 1
                self.tau = self.t
                


            elif self.m > 0:
                if self.t == self.tau:
                    if self.m >= 2:
                        self.updateFrequency = np.floor(para.updateRate / Avar)
                    elif self.m == 1:
                        self.updateFrequency = para.updateRate
                    
                    if self.m % self.updateFrequency == 1:
                        #print(self.timeStamp, self.pulledRec, 'time and pull')
            
                        self.thetas, self.thetas_std = workerTraffic.train(self.timeStamp, self.pulledRec)

                        Avar = logistic(np.mean(self.thetas_std))
                        


                    agent.sample(self.pulledCounts, self.rewards)

                    estimatedTraffic, estimatedFee, cutTime = estimateTraffic.getTraffic(T - self.t, self.thetas, self.thetas_std)
                    #estimatedDensityTraffic, estimatedDensityFee, timeDensity = estimateTraffic.getDensityTraffic(T - self.t, self.thetas, self.thetas_std)

                    f.write(str(self.t) + ',' + str(estimatedTraffic) + ',' + str(estimatedFee) + ',' + str(cutTime) + '\n')
                    
                    
                    pullingArms = agent.get_arms((T - self.t), (B - self.usedBudget), estimatedTraffic, estimatedFee)
                    pullingArmsThisTime = np.array([np.random.binomial(1, p) for p in pullingArms])
            

                    sameArms = np.where(self.pullingArms == pullingArmsThisTime)[0]
                    

                    if self.m == 1:
                        if pullingArmsThisTime[0] == 1:
                            hitid1 = publishHIT1(qualification)
                            if task1Available == 0:
                                task1Available = 1
                            else:
                                print('5ERROR!!!!!!!!!!!!')


                        if pullingArmsThisTime[1] == 1:
                            hitid2 = publishHIT2(qualification)
                            if task2Available == 0:
                                task2Available = 1
                            else:
                                print('6ERROR!!!!!!!!!!!!')

                    elif self.m > 1:

                        #terminate and initialize the batches
                        if np.any(sameArms == 0) == 0:
                            if pullingArmsThisTime[0] == 1:
                                hitid1 = publishHIT1(qualification)
                                if task1Available == 0:
                                    task1Available = 1
                                else:
                                    print('7ERROR!!!!!!!!!!!!')


                            elif pullingArmsThisTime[0] == 0:
                                stopHIT(hitid1)
                                revokeWorkers1(hitid1, qualification)
                                if task1Available == 1:
                                    task1Available = 0
                                else:
                                    print('8ERROR!!!!!!!!!!!!')
                                self.batchTime[0] = 0


                     
                        if np.any(sameArms == 1) == 0:
                            if pullingArmsThisTime[1] == 1:
                                hitid2 = publishHIT2(qualification)
                                if task2Available == 0:
                                    task2Available = 1
                                else:
                                    print('9ERROR!!!!!!!!!!!!')

                            elif pullingArmsThisTime[1] == 0:
                                stopHIT(hitid2)
                                revokeWorkers2(hitid2, qualification)
                                if task2Available == 1:
                                    task2Available = 0
                                else:
                                    print('10ERROR!!!!!!!!!!!!')
                                self.batchTime[1] = 0

                                

                    self.pullingArms = pullingArmsThisTime.copy()

                    self.m += 1


                    batchTime = designBatch.arithmetic(cutTime, T - self.t, pullingArmsThisTime)
                    self.tau += batchTime

                    

                if self.batchTime[0] >= cutTime[0]:
                    stopHIT(hitid1)
                    revokeWorkers1(hitid1, qualification)
                    if task1Available == 1:
                        task1Available = 0
                    else:
                        print('11ERROR!!!!!!!!!!!!')
                    
                    self.batchTime[0] = 0


                    
                    hitid1 = publishHIT1(qualification)
                    if task1Available == 0:
                        task1Available = 1
                    else:
                        print('12ERROR!!!!!!!!!!!!')



                if self.batchTime[1] >= cutTime[1]:
                    stopHIT(hitid2)
                    revokeWorkers2(hitid2, qualification)
                    if task2Available == 1:
                        task2Available = 0
                    else:
                        print('13ERROR!!!!!!!!!!!!')
                    
                    self.batchTime[1] = 0


                    hitid2 = publishHIT2(qualification)
                    if task2Available == 0:
                        task2Available = 1
                    else:
                        print('14ERROR!!!!!!!!!!!!')


                time.sleep(60)
                self.pulledTime += self.pullingArms
                self.batchTime[pullingArmsThisTime == 1] += 1
                
                timeStamp = self.batchTime.copy()
               
                
                workerBefore[timeStamp == 0] = realWorkers[timeStamp == 0]
                workerBefore[timeStamp == 1] = realWorkers[timeStamp == 1]
               


                self.realTime = str(datetime.now().strftime("%H:%M:%S"))
                if np.any(timeStamp > para.batchCreateTime) == 1:
                    
                    realWorkers, rewardThisTime = analyzeResults(qualification)
                    
                    workerDiff = (realWorkers - workerBefore).copy()
                  
                    fee = feeCount(realWorkers)
                    self.usedBudget = fee
                    
                    self.pulledCounts = realWorkers.copy()
                    self.rewards = rewardThisTime.copy()

                    #reform timestamp data for model training
                    timeStamp -= para.batchCreateTime
                    timeStampNoZero = np.where(timeStamp < 0, 0, timeStamp)

                    if np.all(workerDiff >= 0) == 1:
                        self.timeStamp = np.vstack((self.timeStamp, timeStampNoZero))
                        self.pulledRec = np.vstack((self.pulledRec, workerDiff))

                    timeStamp += para.batchCreateTime
                

                
                self.t += 1

                self.recordsPrint()

                print('RESULT OF A BATCH    t: ' + str(self.t) + ' pro: ' + str(self.pullingArms) + ' assign: ' + str(np.sum(self.pulledCounts)) + ' fee: ' + str(self.usedBudget / 100))
                    
                

                if self.t >= T or self.usedBudget >= B:
                
                    #terminate the batch
                    if pullingArmsThisTime[0] == 1:
                        stopHIT(hitid1)
                        revokeWorkers1(hitid1, qualification)
                        if task1Available == 1:
                            task1Available = 0
                        else:
                            print('15ERROR!!!!!!!!!!!!')

                    if pullingArmsThisTime[1] == 1:
                        stopHIT(hitid2)
                        revokeWorkers2(hitid2, qualification)
                        if task2Available == 1:
                            task2Available = 0
                        else:
                            print('16ERROR!!!!!!!!!!!!')
                    
                    print(task1Available, task2Available, 'check task terminal')

                    
                    df = pd.DataFrame({
                        'time1': [stamp[0] for stamp in self.timeStamp],
                        'time2': [stamp[1] for stamp in self.timeStamp],
                        'workers1': [workers[0] for workers in self.pulledRec],
                        'workers2': [workers[1] for workers in self.pulledRec]
                    })
                    
                    df.to_csv('timeTrainingDataLP' + str(datetime.now().day) + '.csv', mode='w')

                    pullingIsFeasible = 0
                    


            
    def recordsPrint(self):
        self.selectedArmsR.append(self.pullingArms)
        self.rewardsR = np.vstack((self.rewardsR, self.rewards))
        self.pulledCountsR = np.vstack((self.pulledCountsR, self.pulledCounts))
        self.pulledTimeR = np.vstack((self.pulledTimeR, self.pulledTime))
        self.BudgetR.append([self.usedBudget])
        self.tR.append([self.t])
        self.meanR.append([str(self.thetas)])
        self.stdR.append([str(self.thetas_std)])
        self.updateFrequencyR.append([self.updateFrequency])
        self.realTimeR.append([self.realTime])

        results = np.hstack((np.array(self.selectedArmsR), self.rewardsR, self.pulledCountsR, self.pulledTimeR, np.array(self.BudgetR), np.array(self.tR), np.array(self.meanR), np.array(self.stdR), np.array(self.updateFrequencyR), np.array(self.realTimeR)))
        df = pd.DataFrame(results)
        #, columns=['Selected Project', 'Cumulative Rewards', 'Number of Assignments', 'Remaining Budget', 'Spent Time'])
        
        df.to_csv('realEXPresultsLP' + str(datetime.now().day) + '.csv', mode='w')




if __name__ == "__main__":

    assignment = assigningProcess()
    assignment.assign(TB_UCBalgorithm)

    parameterFile = open('parameter.py', mode='r')
    parameterNote = parameterFile.read()
    file = open('note' + str(datetime.now().day) + '.txt', mode='w')
    file.write(parameterNote)
    file.close()
                                                                                            

    #+ str(datetime.now()) 