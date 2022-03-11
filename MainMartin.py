# -*- coding: utf-8 -*-
#%%
import NeuralNetwork.NN as nn
import Models.BlackScholesModel as bsm
import MonteCarlo.TimeDiscretization as td
import MonteCarlo.BrownianMotion as bm
import MonteCarlo.EulerSchemeFromProcessModel as ep
from Products.EuropeanOption import *
import Utils.LSMGenerator as lsm

import torch
import matplotlib.pyplot as plt
import numpy as np
#%%
#Number of generated samples
nSamples = 8192*2
#Number of timesteps in monte carlo simulation
dt = 10
#Maturity
T = 1.
#Base sigma
sigma0 = 0.2
#Create LSM dataset of sigmas
sigma = sigma0 + np.random.normal(0,0.025, nSamples)
#Strike
K = 1.
#Dispersion around spot for LSM dataset
d = 0.3
#risk free interest rate
r0 = 0.03
#r = r0 + np.random.normal(0, 0.01, nSamples)

#weights for network training
alpha = 1/(1+nSamples)
beta = 1-alpha

#call or Put option flag
callOrPut = 'call'

#Create LSM dataset of spots
S0 = K* np.exp(-0.5*1.5*sigma0*sigma0+sigma0* np.random.normal(0, 1, nSamples))

ST = np.empty(S0.shape[0])
C = np.empty(S0.shape[0])
#Input data for neural network
X = np.c_[S0, sigma]

greeks = np.zeros((S0.shape[0],2))
product = Option(K, callOrPut)
time = td.TimeDiscretization(0, dt, T/dt)
driver = bm.BrownianMotion(time, nSamples, 1)
driver.generateBM()
model = bsm.BlackScholesModel(sigma0, r0)
#Set additional parameters you want to calculate derivatives for. (For BS model vol and riskFreeRate is possible)
model.setDerivParameters(['vol'])
#Create process object.
process = ep.EulerSchemeFromProcessModel(model,driver,time, product)
for i in range(0, S0.shape[0]):
    #For each path:
    #Set vol
    process.model.vol = torch.tensor(sigma[i])
    #Set riskFreeRate
    #process.model.riskFreeRate = torch.tensor(r[i])
    #set spot
    S0_tensor = torch.tensor(S0[i])
    #returns path, payoff and calculated deltas. Be aware of anti flag (Choose antithetic variates or not)
    S,C[i], delta = process.calculateDerivs(S0_tensor, i,anti = True)
    #pathwise delta
    greeks[i,0] = delta[0]
    #pathwise vega
    greeks[i,1] = delta[1]
    #Terminal value
    ST[i] = S[-1]
    ####The following is for checking against analytical formulas for pathwise vega and pathwise delta. OBS! anti must be set to False
    #z = driver.increments[i].sum()
    #pathwiseVega = np.exp(-r0*T)*(-sigma[i]*T+np.sqrt(T)*z)*ST[i] if ST[i]>K else 0
    #print("vega: " + str(pathwiseVega-delta[1]))
    #pathwiseDelta = (np.exp(-r*T)*ST[i]/S0[i] if ST[i]>K else 0)
    #print("delta: " + str(pathwiseDelta-delta[0]))
    #print(greeks[i,0]-(np.exp(-r*T)*ST[i]/S0[i] if ST[i]>K else 0))
    #C[i] = product.payoff(torch.tensor(S), torch.tensor(r), torch.tensor(T))
    
#Train network on inputs
#inputs are: number of input dimensions, output dimensions, hidden layers, neurons in each layer and if it should be a differential or classic neural network
net = nn.NeuralNet(2, 1, 3, 60, differential=True)
#Give the net its inputs, feature-matrix, payoffs and pathwise greeks.
net.generateData(X, C, greeks)
net.prepare()
net.lambda_j = 1.0 / torch.sqrt((net.dydx ** 2).mean(0)).reshape(1, -1)
#Train the network. It prints its loss for each epoch.
net.train(n_epochs = 100, batch_size=256, alpha=alpha, beta=beta, lr=0.01)

#Test the network on a range of inputs.
S0_test = np.linspace(S0.min(), S0.max(), 100)
sigma_test = np.linspace(0.2, 0.2, 100)
#rho_test
rho_test = np.linspace(r0,r0,100)
X_test = np.c_[S0_test, sigma_test]
y_test, dydx_test = net.predict(X_test, True)

#compare with true Black-Scholes price
truePrice = bsm.C(S0_test, K, T, sigma0, r0)
trueDelta = bsm.delta(S0_test, K, T, sigma0, r0, callOrPut)
trueVega = bsm.vega(S0_test, K, T, sigma0, r0, callOrPut)
trueRho = bsm.rho(S0_test,K,T,sigma0,r0,callOrPut)

plt.figure(figsize=[14,8])
plt.subplot(1, 3, 1)
plt.plot(S0, C, 'o', color='grey', label='Simulated payoffs', alpha = 0.3)
plt.plot(S0_test, y_test, color='red', label='NN approximation')
plt.plot(S0_test, truePrice, color='black', label='Black-Scholes price')
plt.xlabel('S0')
plt.ylabel('C')
plt.xlim(0.4,1.6)
plt.legend()
plt.title('Differential ML - Price approximation')

plt.subplot(1, 3, 2)
plt.plot(S0, greeks[:,0], 'o', color='grey', label='AAD deltas', alpha = 0.3)
plt.plot(S0_test, dydx_test[:,0], color='red', label='NN approximated delta')
plt.plot(S0_test, trueDelta, color='black', label='Black-Scholes delta')
plt.xlabel('S0')
plt.ylabel('delta')
plt.xlim(0.4,1.6)
plt.legend()
plt.title('Differential ML - Delta approximation')

plt.subplot(1, 3, 3)
plt.plot(S0, greeks[:,1], 'o', color='grey', label='AAD vega', alpha = 0.3)
plt.plot(S0_test, dydx_test[:,1], color='red', label='NN approximated vega')
plt.plot(S0_test, trueVega, color='black', label='Black-Scholes vega')
plt.xlabel('S0')
plt.ylabel('vega')
plt.xlim(0.4,1.6)
plt.ylim(-0.5,1.5)
plt.legend()
plt.title('Differential ML - Vega approximation')
plt.show()
#plt.savefig('DiffNNTest3.png')

#Gamma sjov

X_test
#set number of variables
if X_test.ndim == 1:
    nTest = 1
else:
    nTest = X_test.shape[1]
#Test if tensor
if torch.is_tensor(X_test) == False:
    X_test = torch.tensor(X_test).view(X_test.shape[0], nTest).float()

#scale
net.x_std
X_scaled = (X_test - net.x_mean) / net.x_std
X_scaled = X_scaled.float()

#Predict on scaled X
y_scaled = net(X_scaled)

#unscale output from net
y = net.y_mean + net.y_std * y_scaled

X_scaled.requires_grad_()
#predict dydx
y_derivs_scaled = torch.autograd.grad(net(X_scaled), X_scaled, grad_outputs=torch.ones(X_scaled.shape[0], 1), create_graph=True, retain_graph=True, allow_unused=True)
#y_derivs = net.y_std/net.x_std * y_derivs_scaled[0]
gamma = torch.autograd.grad(y_derivs_scaled[0][:,0],X_scaled,grad_outputs=torch.ones(X_scaled.shape[0]))
gamma = net.y_std/net.x_std**2 * gamma[0]
gamma = gamma.detach().numpy()

plt.subplot(1, 3, 3)
#plt.plot(S0, greeks[:,1], 'o', color='grey', label='AAD vega', alpha = 0.3)
plt.plot(S0_test, gamma[:,0], color='red', label='NN approximated gamma')
plt.plot(S0_test,bsm.gamma(S0_test,1,1,0.2,0.03), color='black', label='True gamma')
plt.xlabel('S0')
plt.ylabel('Gamma')
plt.xlim(0.4,2)
#plt.ylim(0,1.5)
plt.legend()
plt.title('Differential ML - Vega approximation')
plt.show()

  