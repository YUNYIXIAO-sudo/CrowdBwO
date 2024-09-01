import numpy as np
import pymc3 as pm
from parameterForAnalysis import para


class workerTraffic():
    def train(countTime, pulledRec, workerset):
        #countTime: [[p11, p12, ..., p1k], 
        # [p21, p22, ..., p2k],
        # ...
        # [pTm1, pTm2, ..., pTmk]]

        #pulledRec: [[p11, p12, ..., p1k], 
        # [p21, p22, ..., p2k],
        # ...
        # [pTm1, pTm2, ..., pTmk]]
        

        parameter = []
        parameter_std = []
        
        

        with pm.Model() as model:
            # Priors
            sigma = pm.HalfNormal('sigma', sd = para.sd[0][workerset])
            theta0 = pm.Normal('theta0', mu = para.mu[0][workerset], sd = para.sd[1][workerset])
            theta1 = pm.Normal('theta1', mu = para.mu[1][workerset], sd = para.sd[2][workerset])
            theta2 = pm.Normal('theta2', mu = para.mu[2][workerset], sd = para.sd[3][workerset])
            theta3 = pm.Normal('theta3', mu = para.mu[3][workerset], sd = para.sd[4][workerset])
           

            x1 = pm.Data('x1_data', countTime)
            y = pm.Data('y_data', pulledRec)

            # Non-linear function
            mu = pm.Deterministic('mu', -1 * np.exp(- theta0 * x1 + theta1) * theta2 + theta3)

            
            '''Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)

            # Inference
            trace = pm.sample(4000, tune=1000, step=pm.Slice(), return_inferencedata=False)
            burnedTrace = trace[1000:]'''

            
            observed = pm.Normal('observed', mu=mu, sd=sigma, observed=y)
            
            mean_field = pm.fit(method='advi', n=50000)
            

        trace = mean_field.sample(2000)
        
        for i in range(4):
            parameter.append(trace['theta' + str(i)].mean(axis=0))
            parameter_std.append(trace['theta' + str(i)].std(axis=0))


        return np.array(parameter), np.array(parameter_std)


    '''def predict(self, batchTime):
        #transform to make the prediction easier.

        with self.model:
        # Set the value of X to the new data and generate posterior predictive samples
            pm.set_data({'x1_data': batchTime, 'x2_data': self.totalPulledCounts})
            ppc = pm.sample_posterior_predictive(self.trace, samples=2000)


        mean_estimatedWorkers = np.mean(ppc['y'], axis=0)
        var_estimatedWorkers = np.var(ppc['y'], axis=0)

        return mean_estimatedWorkers, var_estimatedWorkers
    

    def complexPredict(self, addPulledCounts, batchTime):
        pulledCounts = self.totalPulledCounts + addPulledCounts

        with self.model:
        # Set the value of X to the new data and generate posterior predictive samples
            pm.set_data({'x1_data': batchTime, 'x2_data': pulledCounts})
            ppc = pm.sample_posterior_predictive(self.trace, samples=2000)


        mean_estimatedWorkers = np.mean(ppc['y'], axis=0)
        var_estimatedWorkers = np.var(ppc['y'], axis=0)
        
        return mean_estimatedWorkers, var_estimatedWorkers'''
    

'''
from analyze import times
from matplotlib import pyplot as plt

time_t = []

for time in times:
    time = time.split(' ')[1]
    time = time.split('+')[0]
    time = time.split(':')
    t = 0
    for a in range(len(time)):
        t += int(time[len(time) - a - 1]) * (60 ** a)
    time_t.append(t)

time_t.sort()

start = time_t[0]
time_t = [(time - start) for time in time_t]


dis = 60
stamp = []
allTime = 32 #all time = 32 minutes

for a in range(allTime): 
    for t in range(len(time_t)-1):
        if dis * (a + 1) >= time_t[t] and dis * (a + 1) < time_t[t + 1]:
            stamp.append(t)


stamp = stamp[:11]


if __name__ == "__main__":
    
    time = np.arange(1-2, len(stamp) + 1-2, 1)
    time = time

    time = np.repeat(time[:, np.newaxis], 2, axis=1)

    pulled = np.array(stamp)
    pulled = pulled + 0

    pulled = np.repeat(pulled[:, np.newaxis], 2, axis=1)
    
    mean, std = workerTraffic.train(time, pulled)
    
    x = np.arange(1, 13, 1)
    y = -np.exp(-mean[0][0] * x + mean[1][0]) * mean[2][0] + mean[3][0]

    print(y[5], stamp[5])
    print(np.mean(std))

    
    x = np.arange(0, 10, 1)
    y = -1 * np.exp(-0.3 * x - 1) * 30 + 4
    plt.plot(y)
    plt.plot(range(1-2, len(stamp) + 1-2), stamp)
    plt.show()'''


    




