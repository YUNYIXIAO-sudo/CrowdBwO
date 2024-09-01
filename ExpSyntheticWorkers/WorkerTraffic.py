import numpy as np
import pymc3 as pm
from Parameters import para


class workerTraffic():
    def train(countTime, pulledRec, experiment, workerset):
        #countTime: [[p11, p12, ..., p1k], 
        # [p21, p22, ..., p2k],
        # ...
        # [pTm1, pTm2, ..., pTmk]]

        #pulledRec: [[p11, p12, ..., p1k], 
        # [p21, p22, ..., p2k],
        # ...
        # [pTm1, pTm2, ..., pTmk]]
        '''
        if len(countTime) >= para.trainWindow[experiment]:
            countTime = countTime[-para.trainWindow[experiment]:]
            pulledRec = pulledRec[-para.trainWindow[experiment]:]'''

        parameter = []
        parameter_std = []
        nArms = para.nArms[workerset]

        with pm.Model() as model:
            # Priors
            sigma = pm.HalfNormal('sigma', sd = 10 * np.ones(shape=(nArms,)), shape=(nArms,))
            theta0 = pm.Normal('theta0', mu = 0.05 * np.ones(shape=(nArms,)), sd = 0.01 * np.ones(shape=(nArms,)), shape=(nArms,))
            theta1 = pm.Normal('theta1', mu = 0.5 * np.ones(shape=(nArms,)), sd = 0.1 * np.ones(shape=(nArms,)), shape=(nArms,))
            theta2 = pm.Normal('theta2', mu = 60 * np.ones(shape=(nArms,)), sd = 30 * np.ones(shape=(nArms,)), shape=(nArms,))
            theta3 = pm.Normal('theta3', mu = 80 * np.ones(shape=(nArms,)), sd = 50 * np.ones(shape=(nArms,)), shape=(nArms,))
           

            x1 = pm.Data('x1_data', countTime)
            y = pm.Data('y_data', pulledRec)

            # Non-linear function
            mu = pm.Deterministic('mu', -np.exp(-theta0 * x1 + theta1) * theta2 + theta3)
            #np.exp(-1 * (time - env.batchCreateTime) + 5) + 0 * (100) + 5

            
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


        '''# loss function MSE
        def mse_loss(y_true, y_pred):
            loss = ((y_true - y_pred) ** 2).mean()
          
            return loss

        # Calculate loss for each sample
        losses = []
        for theta0t, theta1t, theta2t in zip(trace['theta0'], trace['theta1'], trace['theta2']):
            y_pred = -1 * np.log(theta0t + theta1t * countTime) + theta2t
            loss = mse_loss(pulledRec, y_pred)
            losses.append(loss)

        # Average the losses
        average_loss = np.mean(np.array(losses))'''

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
from timePlot import stamp
from matplotlib import pyplot as plt


if __name__ == "__main__":
    
    time = np.arange(1, len(stamp) + 1, 1)
    time = time

    time = np.repeat(time[:, np.newaxis], para.nArms[0], axis=1)

    pulled = np.array(stamp)
    pulled = pulled

    pulled = np.repeat(pulled[:, np.newaxis], para.nArms[0], axis=1)
    mean, std = workerTraffic.train(time, pulled, 0)
    
    x = np.arange(1, 100, 1)
    y = -np.exp(-mean[0][0] * x + mean[1][0]) * mean[2][0] + mean[3][0]

    print(y[50], stamp[50], y[60], stamp[60])
    print(np.mean(std))
    plt.plot(y)
    plt.plot(range(1, len(stamp) + 1), stamp)
    plt.show()'''


    

    




