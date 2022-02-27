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
#%%
nSamples = 20000
dt = 10
T = 1.
sigma = 0.2 + np.random.normal(0, 0.025, nSamples)
K = 1.
d = 0.25
r = 0.03

#Generate data
S0 = K + d * np.random.normal(0, 1, nSamples)
S0[S0 < 0] = 0.01
#S0 = np.linspace(0.01,3.5,nSamples)
ST = np.empty(S0.shape[0])
C = np.empty(S0.shape[0])
X = np.c_[S0, sigma]
greeks = np.empty((S0.shape[0], 2))


product = Option(K)
time = td.TimeDiscretization(0, dt, T/dt)
driver = bm.BrownianMotion(time, nSamples, 1)
driver.generateBM()
for i in range(0, S0.shape[0]):
    model = bsm.BlackScholesModel(sigma[i], r)
    model.setDerivParameters(['vol'])
    process = ep.EulerSchemeFromProcessModel(model,driver,time, product)    
    S0_tensor = torch.tensor(S0[i])
    S = process.calculateProcess(S0_tensor, i)
    ST[i] = S[-1]
    derivs = process.calculateDerivs(S0_tensor, i)
    greeks[i, 0] = derivs[0] #Delta
    greeks[i, 1] = derivs[1] #Vega
    C[i] = product.payoff(torch.tensor(S), torch.tensor(r), torch.tensor(T))


#Define and train net
net = nn.NeuralNet(2, 1, 6, 60, differential=True)
net.generateData(X, C, greeks)
net.train(n_epochs = 5, batch_size=100, alpha=0.1, beta=0.9, lr=0.001)

#predict
S0_test = np.linspace(S0.min(), S0.max(), 100)
sigma_test = np.linspace(0.2, 0.2, 100)
X_test = np.c_[S0_test, sigma_test]
y_test, dydx_test = net.predict(X_test, True)

#compare with true Black-Scholes price
truePrice = bsm.C(S0_test, K, T, sigma_test, r)
trueDelta = bsm.delta(S0_test, K, T, sigma_test, r)
trueVega = bsm.vega(S0_test, K, T, sigma_test, r)

plt.figure(figsize=[14,8])
plt.subplot(1, 3, 1)
plt.plot(S0, C, 'o', color='grey', label='Simulated payoffs', alpha = 0.3)
plt.plot(S0_test, y_test, color='red', label='NN approximation')
plt.plot(S0_test, truePrice, color='black', label='Black-Scholes price')
plt.xlabel('S0')
plt.ylabel('C')
plt.legend()
plt.title('Differential ML - Price approximation')

plt.subplot(1, 3, 2)
plt.plot(S0, greeks[:,0], 'o', color='grey', label='AAD deltas', alpha = 0.3)
plt.plot(S0_test, dydx_test[:,0], color='red', label='NN approximated delta')
plt.plot(S0_test, trueDelta, color='black', label='Black-Scholes delta')
plt.xlabel('S0')
plt.ylabel('delta')
plt.legend()
plt.title('Differential ML - Delta approximation')

plt.subplot(1, 3, 3)
plt.plot(S0, greeks[:,1], 'o', color='grey', label='AAD vega', alpha = 0.3)
plt.plot(S0_test, dydx_test[:,1], color='red', label='NN approximated vega')
plt.plot(S0_test, trueVega, color='black', label='Black-Scholes vega')
plt.xlabel('S0')
plt.ylabel('vega')
plt.legend()
plt.title('Differential ML - Vega approximation')
plt.show()
#plt.savefig('DiffNNTest.png')