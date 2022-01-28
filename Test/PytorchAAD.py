#%%
from matplotlib.lines import _LineStyle
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
#%%
def f(my_x, my_y):
    return my_x**2 - my_y**2

#%%
x = torch.tensor(2.).requires_grad_()
y = torch.tensor(4.).requires_grad_()
z = f(x,y)
z.backward()
y.grad

#%%
def genPaths(S0, K, T, sigma, r, nObs, nSteps):
    dt = T/nSteps
    dW = torch.tensor(np.random.normal(0, 1, [nSteps, nObs]))
    
    ST = S0 * torch.cumprod(torch.exp((r-sigma**2/2)*dt+sigma*torch.sqrt(dt)*dW), axis=0)

    return ST

def euroCallPrice(S0, K, T, sigma, r, nObs, nSteps):
    ST = genPaths(S0, K, T, sigma, r, nObs, nSteps)
    payoff = torch.maximum(ST[-1,:] - K, torch.tensor(0))
    C = torch.exp(-r*T) * torch.mean(payoff)

    return C

#True BS delta
def deltaBS(S0, K, T, sigma, r):
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r+(sigma**2)/2) * T )
    return norm.cdf(d1)

#delta Pytorch autograd
def mcDelta(S0):
    S0 = torch.tensor(S0).requires_grad_()
    K = torch.tensor(110.).requires_grad_()
    T = torch.tensor(1.).requires_grad_()
    sigma = torch.tensor(0.12).requires_grad_()
    r = torch.tensor(0.03).requires_grad_()

    C = euroCallPrice(S0, K, T, sigma, r, 10000, 100)  
    C.backward()

    return S0.grad.item()
#%%
S0 = 100.
print("True BS delta = ", deltaBS(S0, 110, 1, 0.12, 0.03))
print("Delta from MC and AAD = ", mcDelta(S0))

#%%
x2 = np.empty(1000)
k=0
for i in np.linspace(50., 150., 1000):
    x2[k] = mcDelta(i)
    k += 1

#%%
x = np.linspace(50, 150, 1000)
plt.figure(figsize=(16,12))
plt.plot(x, deltaBS(x, 110, 1, 0.12, 0.03), color='black', label='True Delta')
plt.plot(x, x2, color='red', linestyle='dotted', label='Delta generated by Pytorch')
plt.legend()
plt.savefig('TorchTest.png')

#%%