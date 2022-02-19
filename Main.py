#%%
import NeuralNetwork.NN as nn
import OptionType.EuroCall as ec
import Models.BlackScholesModel as bsm
import MonteCarlo.TimeDiscretization as td
import MonteCarlo.BrownianMotion as bm
import MonteCarlo.EulerSchemeFromProcessModel as ep

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
#%%
def C_BS(S0, K, T, sigma, r):
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r+(sigma**2)/2) * T )
    d2 = d1 - sigma * np.sqrt(T)

    C = norm.cdf(d1)*S0 - norm.cdf(d2)*K*np.exp(-r*T)
    return C
#%%
nSamples = 50000
dt = 1
sigma = 0.2
K = 1.1
d = 1
r = 0.03

#Generate data
S0 = K + d * np.random.normal(0, 1, nSamples)
#S0 = np.linspace(0,5,100000)

time = td.TimeDiscretization(0, 1, 1)
driver = bm.BrownianMotion(time, nSamples, 1, 0)
driver.generateBM()
model = bsm.BlackScholesModel(S0, 0.2, 0.03)
process = ep.EulerSchemeFromProcessModel(model,driver,time)
process.calculateProcess()
S = process.discreteProcess

n = S.shape[0] - 1

S0 = S[0,:]
ST = S[n,:]

#remove nan
S0 = S0[~np.isnan(S0)]
ST = ST[~np.isnan(ST)]

Call = ec.EuroCall(ST, K, dt, r)
C = Call.payoff()

#Define and train net
net = nn.NeuralNet(1,1,6,20)
net.generateData(S0, C)
net.train()

#predict
S0_test = np.linspace(0, 5, 100)
y_test = net.predict(S0_test)

#compare with true Black-Scholes price
truePrice = C_BS(S0_test, K, 1, sigma, r)

plt.plot(S0, C, 'o', color='grey', label='Simulated payoffs', alpha = 0.3)
plt.plot(S0_test, y_test, color='red', label='NN approximation')
plt.plot(S0_test, truePrice, color='black', label='Black-Scholes price')
plt.legend()
plt.title('Standard ML - price approximation')
plt.show()
#plt.savefig('NNtest.png')