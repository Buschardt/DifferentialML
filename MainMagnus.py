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
nSamples = 8192
dt = 10
T = 1.
sigma = 0.2 + np.random.normal(0, 0.025, nSamples)
K = 1.
d = 0.25
r = 0.03 #+ np.random.normal(0, 0.0055, nSamples)
typeFlg = 'put'
#Generate data
S0 = K + d * np.random.normal(0, 1, nSamples)
S0[S0 < 0] = 0.01
#S0 = np.linspace(0.01,3.5,nSamples)
ST = np.empty(S0.shape[0])
C = np.empty(S0.shape[0])
X = np.c_[S0, sigma]
greeks = np.empty((S0.shape[0], 2))

product = Option(K, typeFlg)
time = td.TimeDiscretization(0, dt, T/dt)
driver = bm.BrownianMotion(time, nSamples, 1)
driver.generateBM()
model = bsm.BlackScholesModel(sigma[0], r)
model.setDerivParameters(['vol']) #'riskFreeRate'
process = ep.EulerSchemeFromProcessModel(model,driver,time, product)

for i in range(0, S0.shape[0]):
    process.model.vol = sigma[i]
    S0_tensor = torch.tensor(S0[i])
    S, C[i], greeks[i, :] = process.calculateDerivs(S0_tensor, i, anti=True)
    ST[i] = S[-1]

#Define and train net
net = nn.NeuralNet(2, 1, 4, 20, differential=True)
net.generateData(X, C, greeks)
net.train(n_epochs = 100, batch_size=256, lr=0.1)

#predict
S0_test = np.linspace(S0.min(), S0.max(), 100)
sigma_test = np.linspace(0.2, 0.2, 100)
#r_test = np.linspace(0.03, 0.03, 100)
X_test = np.c_[S0_test, sigma_test]
y_test, dydx_test = net.predict(X_test, True)

#compare with true Black-Scholes price
truePrice = bsm.P(S0_test, K, T, sigma_test, r)
trueDelta = bsm.delta(S0_test, K, T, sigma_test, r, typeFlg)
trueVega = bsm.vega(S0_test, K, T, sigma_test, r, typeFlg)
#trueVega = bsm.rho(S0_test, K, T, sigma_test, r, 'Call')


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
plt.ylim([-0.5,1.5])
plt.xlabel('S0')
plt.ylabel('Vega')
plt.legend()
plt.title('Differential ML - Vega approximation')

plt.show()
#plt.savefig('./Plots/DiffNNTest3.png')

#%%
plt.plot(net.loss)
#%%
#____________________________________________________________________________
###   Longstaff - Schwartz example   ###
nSamples_LSM = 8192*8
antiFlg = False
K = 40
d = 0.1
r = 0.06
timeToMaturity = 1.
discount = np.exp(-(r/dt)*timeToMaturity)
S_LSM = K *np.exp(d * torch.normal(0, 1, size=(nSamples_LSM,1)))
K_LSM = torch.tensor(K)
sigma_LSM = torch.tensor(0.4)
r_LSM = torch.tensor(r)
dt = 49
T_LSM = torch.tensor([timeToMaturity])

C_LSM = np.empty((S_LSM.shape[0]))
greeks_LSM = np.empty((S_LSM.shape[0],1))
dW = torch.randn(nSamples_LSM, dt+1)
St, Et = ls.genPaths(S_LSM, K_LSM, sigma_LSM, r_LSM, T_LSM, dt, dW,type = typeFlg, anti=antiFlg)
st = St.detach().numpy()
et = Et.detach().numpy()
tp,cont = ls.LSM_train_poly(St, Et,discount)
#indeks = 7
#st[:,-indeks][tp[:,-indeks]==1].max()
#st[:,-indeks][tp[:,-indeks]==0].min()

for i in range(S_LSM.shape[0]):
    Si = S_LSM[i]
    Si.requires_grad_()
    tempC = ls.simpleLSM(Si, K_LSM, sigma_LSM, r_LSM, T_LSM, dt, dW[i][:tp[i]+1],type = typeFlg,anti = antiFlg)
    #tempC = ls.LSM(Si, K_LSM, sigma_LSM, r_LSM, T_LSM, dt, dW[i], w, b,type = typeFlg, anti=True)
    tempgreeks = torch.autograd.grad(tempC, [Si], allow_unused=True)[0]
    C_LSM[i] = tempC.detach().numpy()
    greeks_LSM[i] = tempgreeks.detach().numpy()

#ls.simpleLSM(S_LSM[3], K_LSM, sigma_LSM, r_LSM, T_LSM, dt, dW[3][:tp[3]],type = typeFlg,anti = antiFlg)
#genPaths(S_LSM[0], K_LSM, sigma_LSM, r_LSM, T_LSM, dt, dW[0][:tp[0]+1], type=typeFlg, anti=False,tp = dW[0][:tp[0]+1].nelement())
print('Data generated')

#Define and train net
netLSM = nn.NeuralNet(1, 1, 4, 20, differential=True)
netLSM.generateData(S_LSM.detach().numpy(), C_LSM, greeks_LSM)
netLSM.train(n_epochs = 100, batch_size=512)

S0_test = np.linspace(S_LSM.min(), S_LSM.max(), 100)
sigma_test = np.linspace(0.2, 0.2, 100)
#r_test = np.linspace(0.03, 0.03, 100)
X_test = np.c_[S0_test, sigma_test]

#predict
y_LSM_test, dydx_LSM_test = netLSM.predict(X_test[:,0], gradients=True)

plt.figure(figsize=[14,8])
plt.subplot(1, 2, 1)
plt.plot(S0, C, 'o', color='grey', alpha = 0.3, label='Euro samples')
plt.plot(S_LSM.detach().numpy(), C_LSM, 'o', color='lightblue', alpha = 0.3, label='American samples')
plt.plot(S0_test, y_test, color='red', label='NN - Euro')
plt.plot(S0_test, y_LSM_test, color='navy', label='NN - American')
plt.legend()
plt.title(f'Euro (samples: {nSamples}) vs American option (samples: {nSamples_LSM})')
plt.xlabel('S0')
plt.ylabel('C')

plt.subplot(1, 2, 2)
plt.plot(S0, greeks[:,0], 'o', color='grey', alpha = 0.3, label='Euro delta')
plt.plot(S_LSM.detach().numpy(), greeks_LSM, 'o', color='lightblue', alpha = 0.3, label='American delta')
plt.plot(S0_test, dydx_test[:,0], color='red', label='NN - Euro')
plt.plot(S0_test, dydx_LSM_test, color='navy', label='NN - American')
plt.legend()
plt.title(f'Euro (samples: {nSamples}) vs American option (samples: {nSamples_LSM})')
plt.xlabel('S0')
plt.ylabel('Delta') 
plt.show()





y_LSM_test, dydx_LSM_test = netLSM.predict(np.array([36.0,38.0,40.0,42,44]), gradients=True)
#2year,40sigma
100*(y_LSM_test.flatten()-np.array([8.488,7.669,6.921,6.243,5.622]))/y_LSM_test.flatten()
#1year, 40sigma
100*(y_LSM_test.flatten()-np.array([7.091,6.139,5.308,4.588,3.957]))/y_LSM_test.flatten()
y_LSM_test.flatten()-np.array([7.091,6.139,5.308,4.588,3.957])



####LIDT LØST
def predExerciseBoundary(xTrain,yTrain,xTest,degree = 4):
    coefs = poly.polyfit(xTrain, yTrain, degree)
    print(xTrain,yTrain)
    print(coefs)
    return poly.polyval(xTest,coefs).T

stockmatrix = np.array([[1,1.09,1.08,1.34],
                        [1,1.16,1.26,1.54],
                        [1,1.22,1.07,1.03],
                        [1,0.93,0.97,0.92],
                        [1,1.11,1.56,1.52],
                        [1,0.76,0.77,0.90],
                        [1,0.92,0.84,1.01],
                        [1,0.88,1.22,1.34]])

Et = np.maximum(1.1-stockmatrix,0)
print(Et)
Tt = np.array([3]*Et.shape[0])
Tt = np.where(Et[:, -1]>0,Tt,3)
St = stockmatrix
disc = np.exp(-0.06)
for i in range(3)[::-1]:
    y = Et[:, i+1]
    X = St[:, i]
    continuationValue = predExerciseBoundary(X[Et[:,i]>0],  y[Et[:,i]>0]*disc, X,4)
    print(continuationValue)
    inMoney = np.greater(Et[:,i], 0.)
    Tt = np.where((Et[:, i]>continuationValue)*inMoney,i,Tt)
    Et[:, i] = np.where((Et[:, i]>continuationValue)*inMoney,Et[:, i], 0)
    print(Et)
    del continuationValue
Tt

torch.concat((torch.tensor([0.]),torch.tensor([T/dt]*dt)))

