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
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
import torch


#%%
sampler =scipy.stats.qmc.Halton(5, scramble=True, seed=None)

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
means = np.zeros(5)
corr_mat = torch.Tensor([[1, 0.4, -0.3, 0, 0], [0.4, 1, 0, 0, 0.2], [-0.3, 0, 1, 0, 0], [0, 0, 0, 1, 0.15], [0, 0.2, 0, 0.15, 1]])  
vols = torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2]) 
initial_spots = torch.Tensor([1, 1, 1, 1, 1])
K = torch.tensor(20.0)
r = torch.tensor(0.03)
T = torch.tensor(1)
nul = torch.tensor(0.0)
nSamples = 2**12
cov = torch.matmul(torch.matmul(torch.diag(vols),corr_mat),torch.diag(vols))
sigma= _sigma(vols)
sigmabar = _sigmabar(vols,corr_mat)
f = forward(initial_spots,r,sigma,sigmabar,T,nSpots)
price = basketprice(f,K, sigmabar,T, r)

inc0 =  torch.tensor(multivariate_normal(means, np.eye(5)).rvs(size=nSamples//2)*1.5)
inc1 = torch.tensor(multivariate_normal(means,cov).rvs(size=nSamples//2))
initial_spots = initial_spots*torch.exp(0.25*inc0)
initial_spots=torch.tensor(np.random.uniform(0.4,1.6,size=(4096,5)))
initial_spots = torch.tensor(sampler.random(nSamples//2)*1.2+0.4)


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
outputNeurons = 1
hiddenLayers = 4
hiddenNeurons = 20
lr=0.1


net = nn.NeuralNet(nSpots, outputNeurons, hiddenLayers, hiddenNeurons, differential=True)
net.generateData(initial_spots, samples,greeks)
net.train(n_epochs = 100, batch_size=256, lr=lr,alpha=None,beta=None)
testspot=torch.tensor(np.random.uniform(0.4,40,size=(4096,5)))
predicted = net.predict(testspot,True)
f = forward(testspot,r,sigma,sigmabar,T,nSpots,1) 
targets = basketprice(f, K, sigmabar, T, r)
target_delta = basketDelta(f, K, sigmabar, T, r)

deltas = (predicted[1].prod(axis=1)**(1/5)).numpy()*5*np.exp(r.numpy()-.5*sigma.numpy()+0.5*sigmabar.numpy())
xAxis = testspot.prod(axis=1)**(1/5)
plt.figure(figsize=[16,8])
plt.subplot(1,2,1)
plt.plot(S0,samples,'o',color='grey',label=f'Simulated payoffs', alpha = 0.3)
plt.plot(xAxis,predicted[0],'r.',markersize=2,markerfacecolor='white',label='NN approximation')
plt.plot(xAxis,targets,'.',markersize=2,label='Actual value',color='black')
plt.xlabel('geometric average S0')
plt.ylabel('price')
plt.title('Standard ML - price approximation')
plt.legend()
plt.subplot(1,2,2)
plt.plot(S0,greeks.prod(axis=1)**(1/5)*5*np.exp(r.numpy()-.5*sigma.numpy()+0.5*sigmabar.numpy()),'o',color='grey',label=f'Simulated delta', alpha = 0.3)
plt.plot(xAxis,deltas,'r.',markersize=2,markerfacecolor='white',label='NN approximation')
plt.plot(xAxis,target_delta,'.',markersize=2,label='Actual value',color='black')
plt.xlabel('geometric average S0')
plt.ylabel('delta')
plt.title('Standard ML - delta approximation')
plt.legend()


mseprice=np.sqrt(((predicted[0].reshape(-1)-targets.numpy())**2).mean()*100)
msedelta=np.sqrt(((np.nan_to_num(deltas,0.0)-target_delta)**2).mean()*100)


(greeks[S0>1.35].prod(axis=1)**(1/5)*5).min()
