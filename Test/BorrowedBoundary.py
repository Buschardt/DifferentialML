# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:12:10 2022

@author: magnu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:10:13 2022

@author: magnu
"""


#%%
import NeuralNetwork.NN as nn
import Models.BlackScholesModel as bsm
import MonteCarlo.TimeDiscretization as td
import MonteCarlo.BrownianMotion as bm
import MonteCarlo.EulerSchemeFromProcessModel as ep
from Products.EuropeanOption import Option
import Plots.plots as graph

import matplotlib.pyplot as plt
import torch
import numpy as np
#%%
#HELP function

def finiteDifferenceLSM(S0, K,sigma,r,T,dt,discount,nPaths,eps=0.001,anti = False):
    dW = np.random.normal(0,1,(nPaths, dt + 1))
    return (standardLSM(S0+eps,K,sigma,r,T,dt,discount,nPaths,dW,anti)-standardLSM(S0-eps,K,sigma,r,T,dt,discount,nPaths,dW,anti))/(2*eps)


#Simulates Black-Scholes paths and calculates exercise value at each t
def genPaths(S, K, sigma, r, T, dt, dW, type='call', anti=False,tp = None):
    if len(dW.shape) == 2:
        axis = 1
    else:
        axis = 0
    if tp:
        dts = torch.concat((torch.tensor([0.]),torch.tensor([T/(dt)]*dt)))[:tp]
    else:
        dts = torch.concat((torch.tensor([0.]),torch.tensor([T/(dt)]*(dt))))
    St = S * torch.cumprod(torch.exp((r-sigma**2/2)*dts + sigma*torch.sqrt(dts)*dW), axis=axis)
    if type == 'call':
        Et = torch.maximum(St-K, torch.tensor(0))
    elif type == 'put':
        Et = torch.maximum(K-St, torch.tensor(0))
    if anti == True:
        dW_anti = -1 * dW
        St_anti = S * torch.cumprod(torch.exp((r-sigma**2/2)*dts + sigma*torch.sqrt(dts)*dW_anti), axis=axis)
        if type == 'call':
            Et_anti = torch.maximum(St_anti-K, torch.tensor(0))
        elif type == 'put':
            Et_anti = torch.maximum(K-St_anti, torch.tensor(0))
        return torch.cat((St, St_anti), 0), torch.cat((Et, Et_anti), 0)
    return St, Et


def putPayoff(st,k):
    k = np.array([k])
    return np.maximum(k-st,0)

def createBasisFunctions(x_,polDegree,deriv=0):
    x = [[0]*len(x_)] if deriv else [[1]*len(x_)]
    for i in range(1,polDegree+1):
        if deriv:
            x.append(i*np.array(x_)**(i-1))
        else:
            x.append(np.array(x_)**i)
    return np.array(x).T

def lstsquares(x,y,xTest,degree=3):
    x = createBasisFunctions(x,degree)
    xTest = createBasisFunctions(xTest,degree)
    betas = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
    return np.dot(xTest,betas)



def simpleLSM(S,K,sigma,r,T,dt,dW,type = 'call', anti = False):
    if anti:
        St, Et = genPaths(S, K, sigma, r, T, dt, dW[0], type=type, anti=False,tp = dW[0].nelement())
        St_anti, Et_anti = genPaths(S, K, sigma, r, T, dt, dW[1], type=type, anti=False,tp = dW[1].nelement())
        V = (Et[-1]*np.exp(-r*T*(len(dW[0])-1)/dt) + Et_anti[-1]*np.exp(-r*T*(len(dW[1])-1)/dt)) / 2
        return V
    #if not dW.nelement():
    #    return torch.maximum(K-S,torch.tensor(0)) if type == 'put' else torch.maximum(S-K,torch.tensor(0))
    St, Et = genPaths(S, K, sigma, r, T, dt, dW, type=type, anti=False,tp = dW.nelement())
    return Et[-1]*np.exp(-r*T*(len(dW)-1)/dt)

def standardLSM(S0, K,sigma,r,T,dt,discount,nPaths,dW = None,anti = False):
    if dW is None:
        dW = np.random.normal(0,1,(nPaths, dt + 1))
    if anti:
        dW = np.concatenate((dW,-1*dW))
        nPaths = nPaths*2
    n_excerises = dt + 1
    S0_ = S0
    dts = np.concatenate((np.array([0.]),np.array([T/(dt)]*(dt))))
    S0 = np.array([(S0,)]*nPaths)
    St = S0 * np.cumprod(np.exp((r-sigma**2/2)*dts + sigma*np.sqrt(dts)*dW), axis=1)
    Et = np.maximum(K-St,0)
    exercise = Et[:,-1]#.copy()
    itm = exercise>0
    cashflow = Et[:,-1]
    deltas = np.array([0]*nPaths)
    deltas = np.where(itm,-St[:,-1]/S0_,deltas)
    boundaries = []
    #deltas[itm] = -St[:,-1][itm]/S0_
    
    for i in range(n_excerises-1)[::-1]:
        cashflow = cashflow*discount
        deltas = deltas*discount
        X = St[:, i]#.copy()
        exercise = Et[:,i]#.copy()
        itm = exercise>0
        try:
            continuationValue = lstsquares(X[itm], cashflow[itm], X,degree=3)
            boundary = max(X[itm][(continuationValue[itm]<exercise[itm])])
            boundaries.append(boundary)
            ex_idx = (exercise>continuationValue)*itm
            ex_idx = (X<=boundary)
            cashflow[ex_idx] = exercise[ex_idx]
            deltas = np.where(ex_idx,-St[:,i]/S0_,deltas)
        except:
            boundary = 0
            boundaries.append(boundary)
        
    if anti:
        nPaths = nPaths//2
        pairs = np.array([(cashflow[i]+cashflow[i+nPaths])/2 for i in range(nPaths)])
    else:
        pairs = cashflow
    print('standard dev: ',np.std(pairs)/np.sqrt(nPaths))
    print((cashflow).mean())
    print(deltas.mean())
    return np.array(boundaries)#((cashflow).mean(),deltas.mean()),np.array(boundaries)

def LSM_train_poly(St, Et,discount,K,anti,boundaries):
    n_excerises = St.shape[1]
    Tt = np.array([n_excerises]*Et.shape[0])
    St = St.numpy()
    Et = Et.numpy()
    Tt = np.where(Et[:, -1]>0,Tt,n_excerises)
    cashflow = Et[:,-1]
    boundary_ = []
    for i in range(n_excerises-1)[::-1]:
        cashflow = cashflow*discount
        X = St[:, i]#.copy()
        exercise = Et[:,i]#.copy()
        itm = (exercise>0)
        boundary = boundaries[48-i]
        print(boundary)
        try:
            #continuationValue = lstsquares(X[itm], cashflow[itm], X,degree=4)
            
            #boundary = max(X[itm][(continuationValue[itm]<exercise[itm])])
            #boundary_.append(boundary)
            #ex_idx = (exercise>continuationValue)*itm
            
            ex_idx = (X<=boundary)
            Tt = np.where(ex_idx,i,Tt)
            cashflow[ex_idx] = exercise[ex_idx]
        except:
            pass

    return Tt,cashflow,boundary_#,contValues



#parameters
nSamples = 2**12
dt = 49
T = 1.
sigmaMean = 0.2
sigma = sigmaMean + np.random.normal(0, 0.025, nSamples)
K = 40.
d = 0.2

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
boundaries_={}
###   Longstaff - Schwartz example   ###
nSamples_LSM = nSamples
K_LSM = K#torch.tensor(K)
S_LSM = 40+torch.linspace(-20,40,nSamples)#*np.exp(d * torch.normal(0, 1, size=(nSamples_LSM,1))) #+np.array([np.linspace(-20,20,nSamples)]) #np.exp(d * torch.normal(0, 1, size=(nSamples_LSM,1)))
S_LSM = S_LSM.view(-1,1)
sigma_LSM = torch.tensor(sigmaMean)
r_LSM = torch.tensor(rMean)
T_LSM = torch.tensor([T])
discount = np.exp(-(r_LSM.detach().numpy()/dt)*T_LSM.detach().numpy())

C_LSM = np.empty((S_LSM.shape[0]))
greeks_LSM = np.empty((S_LSM.shape[0],1))

dW = torch.randn(nSamples_LSM, dt + 1)
St, Et = genPaths(S_LSM, K_LSM, sigma_LSM, r_LSM, T_LSM, dt, dW, type=typeFlg, anti=antiFlg)
for i in range(30,50):
    boundaries_[i] = standardLSM(i, K, sigmaMean, rMean, T, dt, discount, nPaths)
#_,boundary44 = standardLSM(44, K, sigmaMean, rMean, T, dt, discount, nPaths)


meanboundary=[]
for i in range(49):
    total = 0
    n = 0
    for key,value in boundaries_.items():
        if value[i]:
            n=n+1
            total = total + value[i]
    if n:       
        meanboundary.append(total/n)
    else:
        meanboundary.append(0)
for key,value in boundaries_.items():
    plt.plot(value)
plt.plot(meanboundary)    
plt.ylim([30,40])
tp,cont,boundaries = LSM_train_poly(St, Et, discount,K,anti=antiFlg,boundaries=meanboundary)
#deepInTheMoney = np.where(St[:,0]<26) 
#dpmoney=St[St[:,0]<26].numpy()
#notSoDeepInTheMoney = np.where(St[:,0]<28)
#tp[deepInTheMoney]
#tp[notSoDeepInTheMoney]

for i in range(S_LSM.shape[0]):
    Si = S_LSM[i]
    Si.requires_grad_()
    if antiFlg:
        dW_temp = (dW[i][:tp[i]+1], -1*dW[i][:tp[ S_LSM.shape[0] + i]+1])
    else:
        dW_temp = dW[i][:tp[i]+1]
    tempC = simpleLSM(Si, K_LSM, sigma_LSM, r_LSM, T_LSM, dt, dW_temp, type=typeFlg, anti=antiFlg)
    tempgreeks = torch.autograd.grad(tempC, [Si], allow_unused=True)[0]
    C_LSM[i] = tempC.detach().numpy()
    greeks_LSM[i] = tempgreeks.detach().numpy()

print('Data generated')

#Define and train net - hiddenNeurons
netLSM = nn.NeuralNet(1, outputNeurons, hiddenLayers, hiddenNeurons, differential=diffML)
netLSM.generateData(S_LSM.detach().numpy(), C_LSM, greeks_LSM)
netLSM.train(n_epochs = epochs, batch_size=batchsize, lr=lr,alpha = None,beta = None)

#predict
lsm_domain = np.linspace(20,70,51)
                        
y_LSM_test, dydx_LSM_test = netLSM.predict(lsm_domain, gradients=True)

#LSM_price = [standardLSM(S0, K_LSM, sigmaMean, rMean, T, dt, discount, nPaths,anti = True) for S0 in lsm_domain]
#prices, deltas = [],[]
#for i in LSM_price:
#    prices.append(i[0])
#    deltas.append(i[1])
#LSM_price = prices
#lsm_delta = deltas

#plot
y = np.c_[C_LSM, greeks_LSM]
y_test_LSM = np.c_[y_LSM_test, dydx_LSM_test]
plotLabels_LSM = ['price', 'delta']
y_true_LSM = np.c_[LSM_price, lsm_delta]#np.c_[y_test[:,0], dydx_test[:,0]]

#graph.plotTests(S_LSM, S0_test, y, y_test_LSM, plotLabels_LSM, y_true=y_true_LSM, model='Euro')
#xlim = 5
#xlim = 0
graph.plotTests(S_LSM[xlim:], lsm_domain[xlim:], y[xlim:], y_test_LSM[xlim:], plotLabels_LSM, y_true=np.c_[LSM_price,lsm_delta][xlim:], model='American',error=True)

y_LSM_test, dydx_LSM_test = netLSM.predict(np.array([36.0,38.0,40.0,42,44]), gradients=True)
100*(y_LSM_test.flatten()-np.array([4.472,3.244,2.313,1.617,1.118]))/y_LSM_test.flatten()
print(y_LSM_test.flatten()-np.array([4.472,3.244,2.313,1.617,1.118]))
print(S_LSM.mean())
netLSM.predict(np.array([20,36.0,38.0,40.0,42,44,70]), gradients=True,sec=True)


