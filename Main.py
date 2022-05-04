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
#parameters
nSamples = 2**13
dt = 49
T = 1.
sigmaMean = 0.2
sigma = sigmaMean + np.random.normal(0, 0.025, nSamples)
K = 40.
d = 0.15
rMean = 0.06
r = rMean + np.random.normal(0, 0.0025, nSamples)
typeFlg = 'put'
antiFlg = True

#Network parameters
epochs = 100
batchsize = 256
lr = 0.1
inputNeurons = 3
outputNeurons = 1
hiddenLayers = 4
hiddenNeurons = 20
diffML = True

#seed
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

#Generate data
#S0 = K * np.exp(d * torch.normal(0, 1, size=(nSamples, 1)))
S0 = torch.linspace(K-K*0.5, K+K*0.5, nSamples).view(-1,1)
#S0[S0 < 0] = 0.01
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
    S, C[i], greeks[i, :]  = process.calculateDerivs(S0_tensor, i, anti=antiFlg)
    ST[i] = S[-1]

#Define and train net
net = nn.NeuralNet(inputNeurons, outputNeurons, hiddenLayers, hiddenNeurons, differential=diffML)
net.generateData(X, C, greeks)
net.train(n_epochs = epochs, batch_size=batchsize, lr=lr)

#predict
S0_test = np.linspace(S0.min(), S0.max(), 100)
sigma_test = np.linspace(sigmaMean, sigmaMean, 100)
r_test = np.linspace(rMean, rMean, 100)
X_test = np.c_[S0_test, sigma_test, r_test]
y_test, dydx_test = net.predict(X_test, True, True)

#compare with true Black-Scholes price
truePrice = bsm.V(S0_test, K, T, sigma_test, r_test, typeFlg)
trueDelta = bsm.delta(S0_test, K, T, sigma_test, r_test, typeFlg)
trueVega = bsm.vega(S0_test, K, T, sigma_test, r_test, typeFlg)
trueRho = bsm.rho(S0_test, K, T, sigma_test, r_test, typeFlg)
trueGamma = bsm.gamma(S0_test, K, T, sigma_test, r_test, typeFlg)

#plot
gamma = np.zeros_like(greeks[:,0])
y = np.c_[C, greeks, gamma]
y_test = np.c_[y_test, dydx_test]
plotLabels = ['price', 'delta', 'vega', 'rho', 'gamma']
y_true = np.c_[truePrice, trueDelta, trueVega, trueRho, trueGamma]
graph.plotTests(S0, S0_test, y, y_test, plotLabels, y_true=y_true, error=True)
#%%
plt.plot(net.loss)
#%%
#____________________________________________________________________________
###   Longstaff - Schwartz example   ###
nSamples_LSM = nSamples
K_LSM = torch.tensor(K)
S_LSM = K_LSM * np.exp(d * torch.normal(0, 1, size=(nSamples_LSM,1)))
sigma_LSM =torch.tensor(np.c_[[sigma]*(dt+1)].T)
r_LSM = torch.tensor(rMean)
T_LSM = torch.tensor([T])
discount = np.exp(-(r_LSM.detach().numpy()/dt)*T_LSM.detach().numpy())

C_LSM = np.empty((S_LSM.shape[0]))
greeks_LSM = np.empty((S_LSM.shape[0], 2))
X = np.c_[S_LSM, sigma_LSM[:,0]]

dW = torch.randn(nSamples_LSM, dt + 1)
St, Et = ls.genPaths(S_LSM, K_LSM, sigma_LSM, r_LSM, T_LSM, dt, dW, type=typeFlg, anti=antiFlg)
tp,cont = ls.LSM_train_poly(St, Et, discount)


for i in range(S_LSM.shape[0]):
    Si = S_LSM[i]
    Si.requires_grad_()
    sigmai = sigma_LSM[i,0]
    sigmai.requires_grad_()
    if antiFlg:
        dW_temp = [dW[i][:tp[i]+1], -1*dW[i][:tp[ S_LSM.shape[0] + i]+1]]
    else:
        dW_temp = dW[i][:tp[i]+1]

    tempC = ls.simpleLSM(Si, K_LSM, sigmai, r_LSM, T_LSM, dt, dW_temp, type=typeFlg, anti=antiFlg)
    tempgreeks = torch.autograd.grad(tempC, [Si, sigmai], allow_unused=True)
    C_LSM[i] = tempC.detach().numpy()
    greeks_LSM[i,:] = np.asarray(tempgreeks)

print('Data generated')

#Define and train net
netLSM = nn.NeuralNet(2, outputNeurons, hiddenLayers, hiddenNeurons, differential=diffML)
netLSM.generateData(X, C_LSM, greeks_LSM)
netLSM.train(n_epochs = epochs, batch_size=batchsize, lr=lr)

#predict
y_LSM_test, dydx_LSM_test = netLSM.predict(X_test[:,:2], gradients=True)

#plot
y = np.c_[C_LSM, greeks_LSM]
y_test_LSM = np.c_[y_LSM_test, dydx_LSM_test]
plotLabels_LSM = ['price', 'delta', 'vega']
y_true_LSM = np.c_[y_test[:,0], dydx_test[:,0], dydx_test[:,1]]

graph.plotTests(S_LSM, S0_test, y, y_test_LSM, plotLabels_LSM, y_true=y_true_LSM, model='Euro')