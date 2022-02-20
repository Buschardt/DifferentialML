#%%
import NeuralNetwork.NN as nn
import Models.BlackScholesModel as bsm
import MonteCarlo.TimeDiscretization as td
import MonteCarlo.BrownianMotion as bm
import MonteCarlo.EulerSchemeFromProcessModel as ep
from Products.EuropeanOption import Option

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
#%%
def C_BS(S0, K, T, sigma, r):
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r+(sigma**2)/2) * T )
    d2 = d1 - sigma * np.sqrt(T)

    C = norm.cdf(d1)*S0 - norm.cdf(d2)*K*np.exp(-r*T)
    return C

def deltaBS(S0, K, T, sigma, r):
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r+(sigma**2)/2) * T )
    return norm.cdf(d1)
#%%
nSamples = 20000
dt = 1
sigma = 0.2
K = 1.1
d = 1
r = 0.03

S0_0 = 0.01
S0_T = 2

#Generate data
#S0 = K + d * np.random.normal(0, 1, nSamples)
S0 = np.linspace(S0_0, S0_T, nSamples)
ST = np.empty(S0.shape[0])
delta = np.empty(S0.shape[0])

product = Option(K)
time = td.TimeDiscretization(0, 1, 1)
driver = bm.BrownianMotion(time, nSamples, 1, 0)
driver.generateBM()
model = bsm.BlackScholesModel(0.2, 0.03)
process = ep.EulerSchemeFromProcessModel(model,driver,time, product)
for i in range(0,S0.shape[0]):
    S0_tensor = torch.tensor(S0[i])
    ST[i] = process.calculateProcess(S0_tensor, i)
    delta[i] = process.calculateDerivs(S0_tensor, torch.tensor(r), torch.tensor(1.),  i)

C = product.payoff(torch.tensor(ST), torch.tensor(r), torch.tensor(1.))


#Define and train net
net = nn.NeuralNet(1,1,6,20, differential=True)
net.generateData(S0, C, delta)
net.train(n_epochs = 10, batch_size=100, alpha=0.25, beta=0.75, lr=0.001)

#predict
S0_test = np.linspace(S0_0, S0_T, 100)
y_test, dydx_test = net.predict(S0_test, True)

#compare with true Black-Scholes price
truePrice = C_BS(S0_test, K, 1, sigma, r)
trueDelta = deltaBS(S0_test, K, 1, sigma, r)

plt.figure(figsize=[14,8])
plt.subplot(1, 2, 1)
plt.plot(S0, C, 'o', color='grey', label='Simulated payoffs', alpha = 0.3)
plt.plot(S0_test, y_test, color='red', label='NN approximation')
plt.plot(S0_test, truePrice, color='black', label='Black-Scholes price')
plt.xlabel('S0')
plt.ylabel('C')
plt.legend()
plt.title('Differential ML - Price approximation')

plt.subplot(1, 2, 2)
plt.plot(S0, delta, 'o', color='grey', label='AAD deltas', alpha = 0.3)
plt.plot(S0_test, dydx_test, color='red', label='NN approximated delta')
plt.plot(S0_test, trueDelta, color='black', label='Black-Scholes delta')
plt.xlabel('S0')
plt.ylabel('delta')
plt.legend()
plt.title('Differential ML - Delta approximation')
plt.show()
#plt.savefig('NNTest.png')