#%%
import NeuralNetwork.NN as nn
import Models.BlackScholesModel as bsm
import MonteCarlo.TimeDiscretization as td
import MonteCarlo.BrownianMotion as bm
import MonteCarlo.EulerSchemeFromProcessModel as ep
from Products.EuropeanOption import Option
import Models.LongstaffSchwartz as ls
import Plots.plots as graph

import matplotlib.pyplot as plt
import torch
import numpy as np
#%%
nSamples = 2**14
dt = 49
T = 1.
sigma = 0.2 + np.random.normal(0, 0.025, nSamples)
K = 40.
d = 0.15
r = 0.06 + np.random.normal(0, 0.0025, nSamples)
typeFlg = 'put'
antiFlg = True

#Generate data
S0 = K * np.exp(d * torch.normal(0, 1, size=(nSamples, 1)))
S0[S0 < 0] = 0.01
#S0 = np.linspace(0.01,3.5,nSamples)
ST = np.empty(S0.shape[0])
C = np.empty(S0.shape[0])
X = np.c_[S0, sigma, r]
greeks = np.empty_like(X)
product = Option(K, typeFlg)
time = td.TimeDiscretization(0, dt, T/dt)
driver = bm.BrownianMotion(time, nSamples, 1)
driver.generateBM()
model = bsm.BlackScholesModel(sigma[0], r)
model.setDerivParameters(['vol', 'riskFreeRate'])
process = ep.EulerSchemeFromProcessModel(model, driver, time, product)

for i in range(0, S0.shape[0]):
    process.model.vol = sigma[i]
    process.model.riskFreeRate = r[i]
    S0_tensor = torch.tensor(S0[i])
    S, C[i], greeks[i, :] = process.calculateDerivs(S0_tensor, i, anti=antiFlg)
    ST[i] = S[-1]

#Define and train net
net = nn.NeuralNet(3, 1, 4, 20, differential=True)
net.generateData(X, C, greeks)
net.train(n_epochs = 100, batch_size=256, lr=0.1)

#predict
S0_test = np.linspace(S0.min(), S0.max(), 100)
sigma_test = np.linspace(0.2, 0.2, 100)
r_test = np.linspace(0.06, 0.06, 100)
X_test = np.c_[S0_test, sigma_test, r_test]
y_test, dydx_test = net.predict(X_test, True)

#compare with true Black-Scholes price
truePrice = bsm.V(S0_test, K, T, sigma_test, r_test, typeFlg)
trueDelta = bsm.delta(S0_test, K, T, sigma_test, r_test, typeFlg)
trueVega = bsm.vega(S0_test, K, T, sigma_test, r_test, typeFlg)
trueRho = bsm.rho(S0_test, K, T, sigma_test, r_test, typeFlg)

#plot
y = np.c_[C, greeks]
y_test = np.c_[y_test, dydx_test]
plotLabels = ['price', 'delta', 'vega', 'rho']
y_true = np.c_[truePrice, trueDelta, trueVega, trueRho]
graph.plotTests(S0, S0_test, y, y_test, plotLabels, y_true=y_true, error=True)

#%%
plt.plot(net.loss)
#%%
#____________________________________________________________________________
###   Longstaff - Schwartz example   ###
nSamples_LSM = 2**14
K_LSM = torch.tensor(40.)
S_LSM = K_LSM * np.exp(d * torch.normal(0, 1, size=(nSamples_LSM,1)))
sigma_LSM = torch.tensor(0.2)
r_LSM = torch.tensor(0.06)
dt = 49
T_LSM = torch.tensor([1.])
discount = np.exp(-(r_LSM.item()/dt)*T_LSM.item())

C_LSM = np.empty((S_LSM.shape[0]))
greeks_LSM = np.empty((S_LSM.shape[0],1))

dW = torch.randn(nSamples_LSM, dt + 1)
St, Et = ls.genPaths(S_LSM, K_LSM, sigma_LSM, r_LSM, T_LSM, dt, dW, type=typeFlg, anti=antiFlg)
tp,cont = ls.LSM_train_poly(St, Et, discount)


for i in range(S_LSM.shape[0]):
    Si = S_LSM[i]
    Si.requires_grad_()
    if antiFlg:
        dW_temp = [dW[i][:tp[i]+1], -1*dW[i][:tp[ S_LSM.shape[0] + i]+1]]
    else:
        dW_temp = dW[i][:tp[i]+1]

    tempC = ls.simpleLSM(Si, K_LSM, sigma_LSM, r_LSM, T_LSM, dt, dW_temp, type=typeFlg, anti=antiFlg)
    tempgreeks = torch.autograd.grad(tempC, [Si], allow_unused=True)[0]
    C_LSM[i] = tempC.detach().numpy()
    greeks_LSM[i] = tempgreeks.detach().numpy()

print('Data generated')

#Define and train net
netLSM = nn.NeuralNet(1, 1, 4, 20, differential=True)
netLSM.generateData(S_LSM.detach().numpy(), C_LSM, greeks_LSM)
netLSM.train(n_epochs = 100, batch_size=256, lr=0.1)

#predict
y_LSM_test, dydx_LSM_test = netLSM.predict(X_test[:,0], gradients=True)

#plot
y = np.c_[C_LSM, greeks_LSM]
y_test_LSM = np.c_[y_LSM_test, dydx_LSM_test]
plotLabels_LSM = ['price', 'delta']
y_true_LSM = np.c_[y_test[:,0], dydx_test[:,0]]

graph.plotTests(S_LSM, S0_test, y, y_test_LSM, plotLabels_LSM, y_true=y_true_LSM, model='Euro')