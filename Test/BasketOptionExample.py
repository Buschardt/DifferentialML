# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:48:18 2022

@author: magnu
"""

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
    return torch.prod(initialSpots,axis=axis)**(1/nSpots)*torch.exp(r-0.5*T*(sigma-sigmabar))

def basketprice(F,K,sigmabar,T,r):
    d1=(np.log(F/K)+0.5*sigmabar*T)/np.sqrt(sigmabar*T)
    d2=d1-np.sqrt(sigmabar*T)
    return np.exp(-r*T)*(F*norm.cdf(d1)-K*norm.cdf(d2))
    
nSpots = 5
corr_mat = torch.Tensor([[1, 0.4, -0.3, 0, 0], [0.4, 1, 0, 0, 0.2], [-0.3, 0, 1, 0, 0], [0, 0, 0, 1, 0.15], [0, 0.2, 0, 0.15, 1]])  
vols = torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2]) 
initial_spots = torch.Tensor([1, 1, 1, 1, 1])
K = torch.tensor(1.0)
r = torch.tensor(0.00)
T = torch.tensor(1)
nul = torch.tensor(0.0)
nSamples = 2**13
cov = torch.matmul(torch.matmul(torch.diag(vols),corr_mat),torch.diag(vols))
sigma= _sigma(vols)
sigmabar = _sigmabar(vols,corr_mat)
f = forward(initial_spots,r,sigma,sigmabar,T,nSpots)
price = basketprice(f,K, sigmabar,T, r)

inc0 =  torch.tensor(multivariate_normal(means, cov_mat).rvs(size=nSamples//2)*1.5)
inc1 = torch.tensor(multivariate_normal(means,cov_mat).rvs(size=nSamples//2))
initial_spots = initial_spots*torch.exp(inc0)
greeks = np.empty((initial_spots.shape))
for i in range(nSamples//2):
    init_spot = initial_spots[i]
    init_spot.requires_grad_()
    term_spot = init_spot*torch.exp((r-0.5*vols*vols)*T)*torch.exp(T*inc1[i])
    term_spot_anti = init_spot*torch.exp((r-0.5*vols*vols)*T)*torch.exp(T*-inc1[i])
    sample = (torch.exp(-r*T)*torch.maximum(term_spot.prod()**(1/5)-K,nul)+torch.exp(-r*T)*torch.maximum(term_spot_anti.prod()**(1/5)-K,nul))/2
    greeks[i,:] = torch.autograd.grad(sample,[init_spot])[0]

termSpots = initial_spots * torch.exp((r-0.5*vols*vols)*T)*torch.exp(T*inc1)
termSpots_anti = initial_spots * torch.exp((r-0.5*vols*vols)*T)*torch.exp(T*-inc1)
samples = (torch.exp(-r*T)*torch.maximum(termSpots.prod(axis=1)**(1/5)-K,nul)+torch.exp(-r*T)*torch.maximum(termSpots_anti.prod(axis=1)**(1/5)-K,nul))/2
S0 = initial_spots.prod(axis=1)**(1/5)



net = nn.NeuralNet(nSpots, outputNeurons, hiddenLayers, hiddenNeurons, differential=True)
net.generateData(initial_spots, samples,greeks)
net.train(n_epochs = 100, batch_size=batchsize, lr=lr)
testspot=torch.tensor(np.random.uniform(0.4,1.5,size=(4096,5)))
predicted = net.predict(testspot)
f = forward(testspot,r,sigma,sigmabar,T,nSpots,1) 
targets = basketprice(f, K, sigmabar, T, r)
xAxis = testspot.prod(axis=1)**(1/5)
plt.figure(figsize=[8,8])
plt.plot(S0,samples,'o')
plt.plot(xAxis,predicted,'r.',markersize=2)
plt.plot(xAxis,targets,'co',markersize=2,markerfacecolor='white')

f = forward(np.array([1,1,1,1,1]),r,sigma,sigmabar,T,nSpots) 
targets = basketprice(f, K, sigmabar, T, r)
net.predict(np.array([[1,1,1,1,1]]))
indices=(S0<1.01) * (S0>0.99)
samples[indices].mean()
