#%%
import NeuralNetwork.NN as nn
import Models.BlackScholesModel as bsm
import MonteCarlo.TimeDiscretization as td
import MonteCarlo.BrownianMotion as bm
import MonteCarlo.EulerSchemeFromProcessModel as ep
from Products.EuropeanOption import Option
import Models.LongstaffSchwartz as ls
import Plots.plots as graph
import scipy
import matplotlib.pyplot as plt
import torch
import numpy as np
#%%
#parameters
nSamples = 2**12
dt = 49
T = 1.
sigmaMean = 0.2
sigma = sigmaMean + np.random.normal(0, 0.025, nSamples)
K = 40.
d = 0.20
rMean = 0.06
r = rMean + np.random.normal(0, 0.0025, nSamples)
typeFlg = 'put'
antiFlg = True
nPaths = 50000

#Network parameters
epochs = 100
batchsize = max(256,nSamples//16)
lr = 0.1
inputNeurons = 1
outputNeurons = 1
hiddenLayers = 4
hiddenNeurons = 20
diffML = True

#Generate data
S0 = torch.linspace(K-K*0.5, K+K*0.5, nSamples).view(-1,1)
#S0[S0 < 0] = 0.01
ST = np.empty(S0.shape[0])
C = np.empty(S0.shape[0])
X = np.c_[S0]
greeks = np.empty_like(X)
product = Option(K, typeFlg)
time = td.TimeDiscretization(0, dt, T/dt)
driver = bm.BrownianMotion(time, nSamples, 1)
driver.generateBM()
model = bsm.BlackScholesModel(sigmaMean, rMean)
model.setDerivParameters(['vol', 'riskFreeRate'])
process = ep.EulerSchemeFromProcessModel(model, driver, time, product)

for i in range(0, S0.shape[0]):
    #process.model.vol = sigma[i]
    #process.model.riskFreeRate = r[i]
    S0_tensor = torch.tensor(S0[i])
    S, C[i], greeks[i, :]  = process.calculateDerivs(S0_tensor, i, anti=antiFlg)
    ST[i] = S[-1]

#Define and train net
net = nn.NeuralNet(inputNeurons, outputNeurons, hiddenLayers, hiddenNeurons, differential=diffML)
net.generateData(X, C, greeks)
net.train(n_epochs = epochs, batch_size=batchsize, lr=lr,alpha=0.01,beta=0.99)

#predict
S0_test = np.linspace(S0.min(), S0.max(), 100)
sigma_test = np.linspace(sigmaMean, sigmaMean, 100)
r_test = np.linspace(rMean, rMean, 100)
X_test = np.c_[S0_test]
y_test, dydx_test = net.predict(X_test, True, True)

#compare with true Black-Scholes price
truePrice = bsm.V(S0_test, K, T, sigmaMean, rMean, typeFlg)
trueDelta = bsm.delta(S0_test, K, T, sigmaMean, rMean, typeFlg)
trueVega = bsm.vega(S0_test, K, T, sigma_test, r_test, typeFlg)
trueRho = bsm.rho(S0_test, K, T, sigma_test, r_test, typeFlg)
trueGamma = bsm.gamma(S0_test, K, T, sigmaMean, rMean, typeFlg)

#plot
gamma = np.zeros_like(greeks[:,0])
y = np.c_[C, greeks, gamma]
y_test = np.c_[y_test, dydx_test]
plotLabels = ['price', 'delta', 'gamma']
y_true = np.c_[truePrice, trueDelta, trueGamma]
graph.plotTests(S0, S0_test, y, y_test, plotLabels, y_true=y_true, error=True)
#%%
plt.plot(net.loss)
#%%
#____________________________________________________________________________
###   Longstaff - Schwartz example   ###
nSamples_LSM = 2**12
d=0.2
K_LSM = K#torch.tensor(K)
sampler =scipy.stats.qmc.Halton(1, scramble=True, seed=None)
S_LSM = torch.tensor(sampler.random(nSamples_LSM)*50+20)
sigma_LSM = torch.tensor(sigmaMean)
r_LSM = torch.tensor(rMean)
T_LSM = torch.tensor([T])
discount = np.exp(-(r_LSM.detach().numpy()/dt)*T_LSM.detach().numpy())
batchsize = max(256,nSamples//16)
C_LSM = np.empty((S_LSM.shape[0]))
greeks_LSM = np.empty((S_LSM.shape[0],1))

dW = torch.randn(nSamples_LSM, dt + 1)
St, Et = ls.genPaths(S_LSM, K_LSM, sigma_LSM, r_LSM, T_LSM, dt, dW, type=typeFlg, anti=antiFlg)
stnumpy = St.numpy()
etnumpy = Et.numpy()
tp,cont,boundaries = ls.LSM_train_poly(St, Et, discount)

for i in range(S_LSM.shape[0]):
    Si = S_LSM[i]
    Si.requires_grad_()
    if antiFlg:
        dW_temp = (dW[i][:tp[i]+1], -1*dW[i][:tp[ S_LSM.shape[0] + i]+1])
    else:
        dW_temp = dW[i][:tp[i]+1]
    
    
    tempC = ls.simpleLSM(Si, K_LSM, sigma_LSM, r_LSM, T_LSM, dt, dW_temp, type=typeFlg, anti=antiFlg)
    tempgreeks = torch.autograd.grad(tempC, [Si], allow_unused=True)[0]
    C_LSM[i] = tempC.detach().numpy()
    greeks_LSM[i] = tempgreeks.detach().numpy()

print('Data generated')

#Define and train net - hiddenNeurons
netLSM = nn.NeuralNet(1, outputNeurons, hiddenLayers, hiddenNeurons, differential=diffML)
netLSM.generateData(S_LSM.detach().numpy(), C_LSM, greeks_LSM)
netLSM.train(n_epochs = epochs, batch_size=batch_size, lr=lr,alpha = None,beta = None)

#predict
lsm_domain = np.linspace(20,70,51)
                        
y_LSM_test, dydx_LSM_test = netLSM.predict(lsm_domain, gradients=True)

LSM_price = [ls.standardLSMwithDelta(S0, K_LSM, sigmaMean, rMean, T, dt, discount, nPaths,anti = True) for S0 in lsm_domain]
prices, deltas = [],[]
for i in LSM_price:
    prices.append(i[0])
    deltas.append(i[1])
LSM_price = prices
lsm_delta = deltas

#plot
y = np.c_[C_LSM, greeks_LSM]
y_test_LSM = np.c_[y_LSM_test, dydx_LSM_test]
plotLabels_LSM = ['price', 'delta']
y_true_LSM = np.c_[LSM_price, lsm_delta]#np.c_[y_test[:,0], dydx_test[:,0]]
mseprice=np.sqrt(((y_LSM_test.flatten()-LSM_price)**2).mean())
msedelta=np.sqrt(((dydx_LSM_test.flatten()-np.array(lsm_delta))**2).mean())
xlim = 0
graph.plotTests(S_LSM[xlim:], lsm_domain[xlim:], y[xlim:], y_test_LSM[xlim:], plotLabels_LSM, y_true=y_true_LSM, model='American',error=True)

y_LSM_test, dydx_LSM_test = netLSM.predict(np.array([36.0,38.0,40.0,42,44]), gradients=True)
100*(y_LSM_test.flatten()-np.array([4.472,3.244,2.313,1.617,1.118]))/y_LSM_test.flatten()
print(y_LSM_test.flatten()-np.array([4.472,3.244,2.313,1.617,1.118]))
print(S_LSM.mean())
netLSM.predict(np.array([100,36.0,38.0,40.0,42,44,70]), gradients=True,sec=True)



