# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:01:29 2022

@author: magnu
"""

from dis import disco
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy.polynomial.polynomial as poly
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import NeuralNetwork.NN as nn
import Models.BlackScholesModel as bsm
import MonteCarlo.TimeDiscretization as td
import MonteCarlo.BrownianMotion as bm
import MonteCarlo.EulerSchemeFromProcessModel as ep
from Products.EuropeanOption import Option
import Models.LongstaffSchwartz as ls
import Plots.plots as graph
from scipy.stats import kurtosis, skew

def createBasisFunctions(x_,sigma_,r_):
    x =[[1]*len(x_)]
    for i in range(1,3):
        x.append(np.array(x_)**i)
    for i in range(1,3):
        x.append(np.array(sigma_)**i)
    for i in range(1,2):
        x.append(np.array(r_)**i)
    
    x.append(np.array(x_)*np.array(sigma_))
    x.append(np.array(r_)*np.array(x_))
    x.append(np.array(r_)*np.array(sigma_))
    #x.append(np.array(x_)*np.array(sigma_)**2)
    #x.append(np.array(x_)**2*np.array(sigma_))
    return np.array(x).T

def lstsquares(x,sigma,r,y,xTest,sigmaTest,rTest):
    x = createBasisFunctions(x,sigma,r)
    xTest = createBasisFunctions(xTest,sigmaTest,rTest)
    betas = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
    return np.dot(xTest,betas)



def predExerciseBoundary(xTrain,yTrain,xTest,degree = 2,basis = 'polynomial'):
    if basis == 'polynomial':
        coefs = poly.polyfit(xTrain, yTrain, degree,rcond=None)
        print(coefs)
    elif basis == 'laguerre':
        coefs = np.polynomial.laguerre.lagfit(xTrain,yTrain,degree,rcond=None)
        print(coefs)
        return np.polynomial.laguerre.lagval(xTest,coefs).T
    elif basis == 'legendre':
        coefs = np.polynomial.legendre.legfit(xTrain,yTrain,degree,rcond=None)
        print(coefs)
        return np.polynomial.legendre.legval(xTest,coefs).T
    return poly.polyval(xTest,coefs).T



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
        return torch.cat((St, St_anti)).reshape(2, -1), torch.cat((Et, Et_anti)).reshape(2, -1)
    return St, Et


def simpleLSM(S,K,sigma,r,T,dt,dW,type = 'call', anti = False):
    if anti:
        St, Et = genPaths(S, K, sigma, r, T, dt, dW[0], type=type, anti=False,tp = dW[0].nelement())
        St_anti, Et_anti = genPaths(S, K, sigma, r, T, dt, dW[1], type=type, anti=False,tp = dW[1].nelement())
        V = (Et[-1]*torch.exp(-r*T*(len(dW[0])-1)/dt) + Et_anti[-1]*torch.exp(-r*T*(len(dW[1])-1)/dt)) / 2
        return V
    #if not dW.nelement():
    #    return torch.maximum(K-S,torch.tensor(0)) if type == 'put' else torch.maximum(S-K,torch.tensor(0))
    St, Et = genPaths(S, K, sigma, r, T, dt, dW, type=type, anti=False,tp = dW.nelement())
    return Et[-1]*torch.exp(-r*T*(len(dW)-1)/dt)
    



def standardLSM(S0, K,sigma,r,T,dt,discount,nPaths,dW = None,anti = False):
    if dW is None:
        dW = np.random.normal(0,1,(nPaths, dt + 1))
    if anti:
        dW = np.concatenate((dW,-1*dW))
        nPaths = nPaths*2
    n_excerises = dt + 1
    dts = np.concatenate((np.array([0.]),np.array([T/(dt)]*(dt))))
    S0 = np.array([(S0,)]*nPaths)
    St = S0 * np.cumprod(np.exp((r-sigma**2/2)*dts + sigma*np.sqrt(dts)*dW), axis=1)
    Et = np.maximum(K-St,0)
    cashflow = Et[:,-1]
    boundaries = []
    for i in range(n_excerises-1)[::-1]:
        cashflow = cashflow*discount
        X = St[:, i]#.copy()
        exercise = Et[:,i]#.copy()
        itm = exercise>0
        try:
            continuationValue = torch.tensor(ls.lstsquares(X[itm], cashflow[itm], X,degree=3))
            boundary = max(X[itm][(continuationValue[itm]<exercise[itm])])
            boundaries.append(boundary)
            ex_idx = (exercise>continuationValue)*itm
            ex_idx = (X<=boundary)
            cashflow[ex_idx] = exercise[ex_idx]
        #print(i)
        #print(len(exercise[ex_idx]))
        except:
            pass
    if anti:
        nPaths = nPaths//2
        pairs = np.array([(cashflow[i]+cashflow[i+nPaths])/2 for i in range(nPaths)])
    else:
        pairs = cashflow
    print('standard dev: ',np.std(pairs)/np.sqrt(nPaths))
    print((cashflow).mean())
    return (cashflow).mean()#contValues#(cashflow).mean(),

def LSM_train_poly(St, Et,discount,sigma,r):
    n_excerises = St.shape[1]
    Tt = np.array([n_excerises]*Et.shape[0])
    St = St.numpy()
    sigma = sigma.numpy()
    r = r.numpy()
    discount = discount.numpy()
    Et = Et.numpy()
    Tt = np.where(Et[:, -1]>0,Tt,n_excerises)
    cashflow = Et[:,-1]
    #contValues = []
    boundary_ = []
    continuationvalues_ = []
    for i in range(n_excerises-1)[::-1]:
        cashflow = cashflow*discount
        X = St[:, i]#.copy()
        exercise = Et[:,i]#.copy()
        itm = exercise>0
        continuationValue = lstsquares(X[itm]/K, sigma[itm],r[itm],cashflow[itm], X/K,sigma,r)
        continuationvalues_.append(continuationValue)
        ex_idx = (exercise>continuationValue)*itm
        Tt = np.where(ex_idx,i,Tt)
        cashflow[ex_idx] = exercise[ex_idx]
    return Tt,cashflow,boundary_,continuationvalues_#,contValues

nSamples_LSM = 2**13
epochs = 100
batchsize = max(256,nSamples_LSM//16)
lr = 0.1
inputNeurons = 3
outputNeurons = 1
hiddenLayers = 4
hiddenNeurons = 20
diffML = True
sampler =scipy.stats.qmc.Halton(3, scramble=True, seed=None)
antiFlg=True
typeFlg='put'
a=sampler.random(nSamples_LSM)
d=0.2
dt = 49
T = 1.
sigmaMean = 0.2
K = 40.
rMean = 0.06
K_LSM = torch.tensor(K)
S_LSM = torch.tensor(a[:,0]*(55-25)+25)
sigma_LSM = torch.tensor(a[:,1]*0.3+0.15)

r_LSM = torch.tensor(a[:,2]*(0.1-0.02)+0.02)
X = np.c_[S_LSM, sigma_LSM,r_LSM]
T_LSM = torch.tensor([T])
discount = torch.tensor(np.exp(-(r_LSM.detach().numpy()/dt)*T_LSM.detach().numpy()))
batchsize = max(256,nSamples_LSM//16)
C_LSM = np.empty((S_LSM.shape[0]))
greeks_LSM = np.empty((S_LSM.shape[0],inputNeurons))
dW = torch.randn(nSamples_LSM, dt + 1)
S = torch.empty(dW.shape[0]*2,dW.shape[1])
E = torch.empty(dW.shape[0]*2,dW.shape[1])

for i in range(nSamples_LSM):
    St, Et = genPaths(S_LSM[i], K_LSM, sigma_LSM[i], r_LSM[i], T_LSM, dt, dW[i], type='put', anti=True)
    S[i] = St[0]
    S[i+nSamples_LSM]=St[1]
    E[i] = Et[0]
    E[i+nSamples_LSM]=Et[1]

St = S
Et = E
stnumpy = St.numpy()
etnumpy = Et.numpy()
dwnumpy = dW.numpy()
sigmanumpy = sigma_LSM.numpy()
rnumpy = r_LSM.numpy()
sigma_LSM_long=torch.concat((sigma_LSM,sigma_LSM))
discount_long = torch.concat((discount,discount))
r_LSM_long = torch.concat((r_LSM,r_LSM))
tp,cont,boundaries,contValues = LSM_train_poly(St, Et, discount_long,sigma_LSM_long,r_LSM_long)

for i in range(S_LSM.shape[0]):
    Si = S_LSM[i]
    Si.requires_grad_()
    sigmai = sigma_LSM[i]
    sigmai.requires_grad_()
    ri = r_LSM[i]
    ri.requires_grad_()
    if antiFlg:
        dW_temp = (dW[i][:tp[i]+1], -1*dW[i][:tp[ S_LSM.shape[0] + i]+1])
    else:
        dW_temp = dW[i][:tp[i]+1]
    
    
    tempC = simpleLSM(Si, K_LSM, sigmai, ri, T_LSM, dt, dW_temp, type=typeFlg, anti=antiFlg)
    tempgreeks = torch.autograd.grad(tempC, [Si,sigmai,ri], allow_unused=True)
    C_LSM[i] = tempC.detach().numpy()
    greeks_LSM[i,:] = np.asarray(tempgreeks)

print('Data generated')

#Define and train net - hiddenNeurons
netLSM = nn.NeuralNet(inputNeurons, outputNeurons, hiddenLayers, hiddenNeurons, differential=diffML)
netLSM.generateData(X, C_LSM, greeks_LSM)
netLSM.train(n_epochs = epochs, batch_size=batchsize, lr=lr,alpha = None,beta = None)




#a=sampler.random(100)
#testX = np.empty((100,3))
#testX[:,0] = a[:,0]*(50-30)+30
#testX[:,1] = a[:,1]*(.4-.2)+.2
#testX[:,2] = a[:,2]*(0.1-.02)+0.02
#LSM_price = [standardLSM(i[0], K_LSM, i[1], rMean, T, dt, discount, 50000,anti = True) for i in testX]
#LSM_price = []
#k=0
#for i in testX:
#    print(i[0],i[1],i[2])
#    l = standardLSM(i[0], K_LSM, i[1], i[2], T, dt, discount[k], 50000,anti = True)
#    print(l)
#    LSM_price.append(l)
#    k=k+1

standardLSM(40, K_LSM, 0.4, rMean, T, dt, discount, 50000,anti = True)
abc=(S_LSM.numpy()<44.5) * (S_LSM.numpy()>42.5)
cba=(sigma_LSM.numpy()<0.235)*(sigma_LSM.numpy()>0.219)
dec = abc*cba
C_LSM[dec].mean()

errors = netLSM.predict(testX).flatten()-np.array(LSM_price)

netLSM.predict(np.array([[36,.2,0.06],[38,.2,0.06],[40,0.2,0.06],[42,0.2,0.06],[44,0.2,0.06],[36,.4,0.06],[38,.4,0.06],[40,0.4,0.06],[42,0.4,0.06],[44,0.4,0.06]]))
plt.figure(figsize=[12,6])
plt.plot(testX[:,0],[0]*100,color='black')
plt.plot(testX[:,0],errors,'r.',markersize=4,markerfacecolor='white',label = 'errors')
plt.ylim([-1,1])
plt.xlabel('S0')
plt.ylabel('Error')
plt.title('Differential ML - price error')

RMSE_i = np.sqrt((errors**2).mean())
RMSE_i_format = "{:.6f}".format(RMSE_i)
plt.plot([], [], ' ', label=f'RMSE: {RMSE_i_format}')
plt.legend()

netLSM.predict(np.array([[43.56454,0.22736]]))

#predict
lsm_domain = np.linspace(20,70,51)
                        
y_LSM_test, dydx_LSM_test = netLSM.predict(lsm_domain, gradients=True)


#LSM_price = [ls.standardLSM(S0, K_LSM, sigmaMean, rMean, T, dt, discount, nPaths,anti = True) for S0 in lsm_domain]
#lsm_delta = [ls.finiteDifferenceLSM(S0, K_LSM, sigmaMean, rMean, T, dt, discount, nPaths,eps = 1e-1) for S0 in lsm_domain]
#lsm_delta[-1]=0

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
netLSM.predict(np.array([100,36.0,38.0,40.0,42,44,70]), gradients=True,sec=True)