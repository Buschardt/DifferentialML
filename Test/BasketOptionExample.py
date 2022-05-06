# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:48:18 2022

@author: magnu
"""

import NeuralNetwork.NN as nn
import Models.BlackScholesModel as bsm
import MonteCarlo.TimeDiscretization as td
import MonteCarlo.BrownianMotion as bm
import MonteCarlo.EulerSchemeFromProcessModel as ep
from Products.EuropeanOption import Option
import Plots.plots as graph

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
import torch

means = np.zeros(5)
corr_mat = np.matrix([[1, 0.4, -0.3, 0, 0], [0.4, 1, 0, 0, 0.2], [-0.3, 0, 1, 0, 0], [0, 0, 0, 1, 0.15], [0, 0.2, 0, 0.15, 1]])  
vols = np.array([0.2, 0.2, 0.2, 0.2, 0.2]) 
initial_spots = np.array([1, 1, 1, 1, 1])

cov_mat = np.diag(vols).dot(corr_mat).dot(np.diag(vols))
dw=np.random.normal(0,1,(50000, 5))

initial_spots = np.array([100., 100., 100., 100., 100.])
initial_spots = np.array([1, 1, 1, 1, 1])
tte = 1.0
strike = K
seed = 0
num_paths = 50000

results = []
rng = multivariate_normal(means, cov_mat).rvs(size=num_paths)
asd = np.cov(rng.T)

for i in range(num_paths):
    rns = rng[i]
    final_spots = initial_spots * np.exp(-0.5*vols*vols*tte) * np.exp(tte * rns)
    results.append(final_spots)

df = pd.DataFrame(results)
df['payoff'] = ((df.prod(axis=1) **(1/ 5)) - strike).clip(0)

df['payoff'].mean()


mod_vol_1 = (vols ** 2).mean()
mod_vol_2 = vols.dot(corr_mat).dot(vols) / len(vols)**2

mod_fwd = np.product(initial_spots)**(1/len(vols)) * np.exp(-0.5*tte*(mod_vol_1 - mod_vol_2))

d_plus = (np.log(mod_fwd / strike) + 0.5 * mod_vol_2 * tte) / np.sqrt(mod_vol_2 * tte)
d_minus = d_plus - np.sqrt(mod_vol_2 * tte)

mod_fwd * norm.cdf(d_plus) - strike * norm.cdf(d_minus)


mu = np.array([-0.0346,-0.017357,0.001105,0.008354])



#%%


def _sigma(vols):
    return (vols**2).mean()

def _sigmabar(vols,corrmatrix):
    return torch.matmul(torch.matmul(vols,corrmatrix),vols)/len(vols)**2

def forward(initialSpots,r,sigma,sigmabar,T,nSpots,axis = 0):
    return torch.prod(initialSpots,axis=axis)**(1/nSpots)*torch.exp(r*T-0.5*T*(sigma-sigmabar))

def basketprice(F,K,sigmabar,T,r):
    d1=(np.log(F/K)+0.5*sigmabar*T)/np.sqrt(sigmabar*T)
    d2=d1-np.sqrt(sigmabar*T)
    return np.exp(-r*T)*(F*norm.cdf(d1)-K*norm.cdf(d2))

def basketDelta(F,K,sigmabar,T,r):
    return norm.cdf((np.log(F/K)+0.5*sigmabar*T)/np.sqrt(sigmabar*T))
    
nSpots = 5
corr_mat = torch.Tensor([[1, 0.4, -0.3, 0, 0], [0.4, 1, 0, 0, 0.2], [-0.3, 0, 1, 0, 0], [0, 0, 0, 1, 0.15], [0, 0.2, 0, 0.15, 1]])  
vols = torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2]) 
initial_spots = torch.Tensor([1, 1, 1, 1, 1])
K = torch.tensor(1.0)
r = torch.tensor(0.03)
T = torch.tensor(1)
nul = torch.tensor(0.0)
nSamples = 2**13
cov = torch.matmul(torch.matmul(torch.diag(vols),corr_mat),torch.diag(vols))
sigma= _sigma(vols)
sigmabar = _sigmabar(vols,corr_mat)
f = forward(initial_spots,r,sigma,sigmabar,T,nSpots)
price = basketprice(f,K, sigmabar,T, r)

inc0 =  torch.tensor(multivariate_normal(means, np.eye(5)).rvs(size=nSamples//2)*1.5)
inc1 = torch.tensor(multivariate_normal(means,cov_mat).rvs(size=nSamples//2))
initial_spots = initial_spots*torch.exp(0.25*inc0)
torch.tensor(np.random.uniform(0.4,1.6,size=(4096,5)))
greeks = np.zeros((initial_spots.shape))
for i in range(nSamples//2):
    init_spot = initial_spots[i]
    init_spot.requires_grad_()
    term_spot = init_spot*torch.exp((r-0.5*vols*vols)*T+torch.sqrt(T)*inc1[i])
    term_spot_anti = init_spot*torch.exp((r-0.5*vols*vols)*T+torch.sqrt(T)*-inc1[i])
    sample = torch.exp(-r*T)*(torch.maximum(term_spot.prod()**(1/5)-K,nul)+torch.maximum(term_spot_anti.prod()**(1/5)-K,nul))/2
    greeks[i,:] = torch.autograd.grad(sample,[init_spot],allow_unused=True)[0]
termSpots = initial_spots * torch.exp((r-0.5*vols*vols)*T)*torch.exp(T*inc1)
termSpots_anti = initial_spots * torch.exp((r-0.5*vols*vols)*T)*torch.exp(T*-inc1)
samples = torch.exp(-r*T)*(torch.maximum(termSpots.prod(axis=1)**(1/5)-K,nul)+torch.maximum(termSpots_anti.prod(axis=1)**(1/5)-K,nul))/2
S0 = initial_spots.prod(axis=1)**(1/5)

net = nn.NeuralNet(nSpots, outputNeurons, hiddenLayers, hiddenNeurons, differential=True)
net.generateData(initial_spots, samples,greeks)
net.train(n_epochs = 100, batch_size=256, lr=lr,alpha=1,beta=0)
testspot=torch.tensor(np.random.uniform(0.4,1.6,size=(4096,5)))
predicted = net.predict(testspot,True)
f = forward(testspot,r,sigma,sigmabar,T,nSpots,1) 
targets = basketprice(f, K, sigmabar, T, r)
target_delta = basketDelta(f, K, sigmabar, T, r)

deltas = (predicted[1].prod(axis=1)**(1/5)).numpy()*5*np.exp(r.numpy()-sigmabar.numpy())
xAxis = testspot.prod(axis=1)**(1/5)
plt.figure(figsize=[16,8])
plt.subplot(1,2,1)
plt.plot(S0,samples,'o',color='grey',label=f'Simulated payoffs', alpha = 0.3)
plt.plot(xAxis,predicted[0],'r.',markersize=2,markerfacecolor='white',label='NN approximation')
plt.plot(xAxis,targets,'.',markersize=2,label='Actual value',color='black')
plt.xlabel('geometric average S0')
plt.ylabel('price')
plt.xlim([0.4,1.4]) 
plt.title('Differential ML - price approximation')
plt.legend()
plt.subplot(1,2,2)
plt.plot(S0,greeks.prod(axis=1)**(1/5)*5,'o',color='grey',label=f'Simulated delta', alpha = 0.3)
plt.plot(xAxis,deltas,'r.',markersize=2,markerfacecolor='white',label='NN approximation')
plt.plot(xAxis,target_delta,'.',markersize=2,label='Actual value',color='black')
plt.xlabel('geometric average S0')
plt.ylabel('delta')
plt.xlim([0.4,1.4]) 
plt.title('Differential ML - delta approximation')
plt.legend()


mseprice=np.sqrt(((predicted[0].reshape(-1)-targets.numpy())**2).mean()*100)
msedelta=np.sqrt(((np.nan_to_num(deltas,0.0)-target_delta)**2).mean()*100)


(greeks[S0>1.35].prod(axis=1)**(1/5)*5).min()
