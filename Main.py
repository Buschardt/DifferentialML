#%%
import NeuralNetwork.NN as nn
import Models.BlackScholesModel as bsm
import MonteCarlo.TimeDiscretization as td
import MonteCarlo.BrownianMotion as bm
import MonteCarlo.EulerSchemeFromProcessModel as ep
from Products.EuropeanOption import Option
import Models.LongstaffSchwartz as ls

import torch
import matplotlib.pyplot as plt
import numpy as np
#%%
nSamples = 12500
dt = 10
T = 1.
sigma = 0.2 + np.random.normal(0, 0.025, nSamples)
K = 1.
d = 0.25
r = 0.03 #+ np.random.normal(0, 0.0025, nSamples)

#Generate data
S0 = K + d * np.random.normal(0, 1, nSamples)
S0[S0 < 0] = 0.01
#S0 = np.linspace(0.01,3.5,nSamples)
ST = np.empty(S0.shape[0])
C = np.empty(S0.shape[0])
X = np.c_[S0, sigma]
greeks = np.empty((S0.shape[0], 2))

product = Option(K, 'call')
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
    greeks[i, :] = process.calculateDerivs(S0_tensor, i)
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
trueDelta = bsm.delta(S0_test, K, T, sigma_test, r, 'Call')
trueVega = bsm.vega(S0_test, K, T, sigma_test, r, 'Call')

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
#plt.savefig('./Plots/DiffNNTest3.png')

#%%
#____________________________________________________________________________
###   Longstaff - Schwartz example   ###

nSamples_LSM = 1000
S_LSM = K + d * torch.normal(0, 1, size=(nSamples_LSM, 1))
K_LSM = torch.tensor(1.)
sigma_LSM = torch.tensor(0.2)
r_LSM = torch.tensor(0.03)
dts_LSM = torch.tensor([1., 1., 1., 1.]).view(1,4)

C_LSM = torch.empty((S_LSM.shape[0]))
greeks_LSM = torch.empty((S_LSM.shape[0]))
St_LSM = torch.empty((S_LSM.shape[0], 4))
for i in range(S_LSM.shape[0]):
    if i % 100 == 0:
        print(i, '/', nSamples_LSM)
    Si = S_LSM[i]
    dW = torch.randn(1,4)
    St, Et = ls.genPaths(Si, K_LSM, sigma_LSM, r_LSM, dts_LSM, dW)
    w, b = ls.LSM_train(St, Et)
    Si.requires_grad_()
    C_LSM[i] = ls.LSM(Si, K_LSM, sigma_LSM, r_LSM, dts_LSM, dW, w, b)
    greeks_LSM[i] = torch.autograd.grad(C_LSM[i], [Si], allow_unused=True)[0]
    
print('Data generated')

#Define and train net
netLSM = nn.NeuralNet(1, 1, 6, 60, differential=True)
netLSM.generateData(S_LSM.detach(), C_LSM.detach(), greeks_LSM)
netLSM.prepare()
netLSM.lambda_j = 1.0 / torch.sqrt((netLSM.dydx ** 2).mean(0)).reshape(1, -1)
netLSM.train(n_epochs = 5, batch_size=10, alpha=0.1, beta=0.9, lr=0.001)

#predict
y_LSM_test, dydx_LSM_test = netLSM.predict(S0_test, gradients=True)

plt.figure(figsize=[14,8])
plt.subplot(1, 2, 1)
plt.plot(S0, C, 'o', color='grey', alpha = 0.3, label='Euro samples')
plt.plot(S_LSM.detach().numpy(), C_LSM.detach().numpy(), 'o', color='lightblue', alpha = 0.3, label='American samples')
plt.plot(S0_test, y_test, color='red', label='NN - Euro')
plt.plot(S0_test, y_LSM_test, color='navy', label='NN - American')
plt.legend()
plt.title(f'Euro (samples: {nSamples}) vs American option (samples: {nSamples_LSM})')
plt.xlabel('S0')
plt.ylabel('C')

plt.subplot(1, 2, 2)
plt.plot(S0, greeks[:,0], 'o', color='grey', alpha = 0.3, label='Euro delta')
plt.plot(S_LSM.detach().numpy(), greeks_LSM.detach().numpy(), 'o', color='lightblue', alpha = 0.3, label='American delta')
plt.plot(S0_test, dydx_test[:,0], color='red', label='NN - Euro')
plt.plot(S0_test, dydx_LSM_test, color='navy', label='NN - American')
plt.legend()
plt.title(f'Euro (samples: {nSamples}) vs American option (samples: {nSamples_LSM})')
plt.xlabel('S0')
plt.ylabel('Delta')
plt.show()
#plt.savefig('./Plots/LongstaffvsEuro')