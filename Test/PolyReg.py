#%%
import Models.BlackScholesModel as bsm
import MonteCarlo.TimeDiscretization as td
import MonteCarlo.BrownianMotion as bm
import MonteCarlo.EulerSchemeFromProcessModel as ep
from Products.EuropeanOption import Option

import matplotlib.pyplot as plt
import torch
import numpy as np
import random

def genDesignMatrix(S0, n, p):
    #Generates design matrix for linear regression
    X = np.ones(n)
    for i in range(1, p+1):
        X = np.c_[X, S0**i]
    return X

def linearRegr(X, Y, C, D, w, deltaReg):
    #This function performs regression
    if deltaReg == False:
        try:
            return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),C) #Standard OLS
        except:
            return np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),C) #Standard OLS
    #Else with delta regularization
    try:
        return np.dot(np.linalg.inv(w*np.dot(X.T,X) + (1-w)*np.dot(Y.T,Y)),(w*np.dot(X.T,C) + (1-w) * np.dot(Y.T,D)))
    except:
        return np.dot(np.linalg.pinv(w*np.dot(X.T,X) + (1-w)*np.dot(Y.T,Y)),(w*np.dot(X.T,C) + (1-w) * np.dot(Y.T,D)))

#Estimate delta from regression
def estDelta(x, theta, p):
    #Derivative of f_theta
    sum = 0
    for i in range(1, p+1):
        sum = sum + i * theta[i] * x**(i-1)
    return sum

#price estimation from regression
def f_theta(x, theta, p):
    #Call value estimated by linear regression
    sum = 0
    for i in range(0, p+1):
        sum = sum + theta[i]*x**i
    return sum
def genY(S, n, p):
    #Generates Y matrix for Delta regularized linear regression
    Y = np.c_[np.zeros(n), np.ones(n)]
    for i in range(2,p+1):
        Y = np.c_[Y, i * S**(i-1)]
    return Y
#%%
#parameters
nSamples = 2**12
dt = 1
T = 1.
sigmaMean = 0.2
sigma = sigmaMean #+ np.random.normal(0, 0.025, nSamples)
K = 40.
d = 0.15
rMean = 0.06
r = rMean #+ np.random.normal(0, 0.0025, nSamples)
typeFlg = 'call'
antiFlg = True

#Seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

#Generate data
#S0 = K * np.exp(d * torch.normal(0, 1, size=(nSamples, 1)))
S0 = torch.linspace(22,67,nSamples).view(-1,1)
S0[S0 < 0] = 0.01
ST = np.empty(S0.shape[0])
C = np.empty(S0.shape[0])
X = S0
greeks = np.empty_like(X)
product = Option(K, typeFlg)
time = td.TimeDiscretization(0, dt, T/dt)
driver = bm.BrownianMotion(time, nSamples, 1)
driver.generateBM()
model = bsm.BlackScholesModel(sigma, r)
#model.setDerivParameters(['vol', 'riskFreeRate'])
process = ep.EulerSchemeFromProcessModel(model, driver, time, product)

for i in range(0, S0.shape[0]):
    #process.model.vol = sigma[i]
    #process.model.riskFreeRate = r[i]
    S0_tensor = torch.tensor(S0[i])
    S, C[i], greeks[i] = process.calculateDerivs(S0_tensor, i, anti=antiFlg)
    ST[i] = S[-1]


n = nSamples
p = 7
X = genDesignMatrix(S0, n, p)
Y = genY(S0, n, p)
theta = linearRegr(X, Y, C.reshape(-1,1), greeks, 0.0001, False)

#Generate test data
x = np.linspace(S0.min(), S0.max(), 1000).reshape(-1,1)
X_test = np.c_[x, x**2, x**3]
C_test = f_theta(X_test, theta, p)
delta_test = estDelta(X_test, theta, p)

truePrice = bsm.V(x, K, T, sigma, r, typeFlg)
trueDelta = bsm.delta(x, K, T, sigma, r, typeFlg)

plt.figure(figsize=[10,6])
plt.subplot(1,2,1)
plt.plot(S0, C, 'o', color='grey', alpha=0.3, label='Simulated payoffs')
plt.plot(X_test[:,0], C_test[:,0], color='red', linewidth=1, label='Poly. reg. approximation')
plt.plot(X_test[:,0], truePrice, color='black', linewidth=1, label='Black-Scholes price')
plt.xlabel('S0')
plt.ylabel('C')
plt.legend()

plt.subplot(1,2,2)
plt.plot(S0, greeks, 'o', color='grey', alpha=0.3)
plt.plot(X_test[:,0], delta_test[:,0], color='red', linewidth=1, label='Approximated delta')
plt.plot(X_test[:,0], trueDelta, color='black', linewidth=1, label='Black-Scholes delta')
plt.ylim([-0.05, 1.15])
plt.xlabel('S0')
plt.ylabel('Delta')
plt.legend()
plt.tight_layout()

plt.show()

error = C_test[:,0].reshape(-1,1) - truePrice
RMSE = np.sqrt((error**2).mean())
print('RMSE price = ', RMSE)
error_delta = delta_test[:,0].reshape(-1,1) - trueDelta
RMSE_delta = np.sqrt((error_delta**2).mean())
print('RMSE delta = ', RMSE_delta)
#%%
plt.plot(error.T)