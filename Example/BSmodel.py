# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:04:54 2022

@author: magnu
"""
import os
os.chdir(r'C:\Users\magnu\DifferentialML')
from Models.BlackScholesModel import BlackScholesModel
from MonteCarlo.TimeDiscretization import TimeDiscretization
from MonteCarlo.BrownianMotion import BrownianMotion
from MonteCarlo.EulerSchemeFromProcessModel import EulerSchemeFromProcessModel
from Products.EuropeanOption import Option

product = Option(100)
time = TimeDiscretization(0,10,0.1)
driver = BrownianMotion(time,10000,1)
driver.generateBM()
model = BlackScholesModel(0.2,0.03)
process = EulerSchemeFromProcessModel(model,driver,time,product)
product.payoff(process.calculateProcess(111.,5)[-1]).backward()
process.initialState.grad
model.vol.grad.grad_fn
model.vol.grad.zero_()
for i in range(100):
    product.payoff(process.calculateProcess(100.,i)[-1]).backward(retain_graph = True)


model.initialValue.grad/10
process.calculateProcess()
a = process.discreteProcess
b=torch.exp(-model.riskFreeRate)*torch.mean(torch.maximum(a[-1]-100.,torch.tensor(0)))
#torch.autograd.grad(b,[model.initialState],torch.ones(5))
b.backward()
model.initialValue.grad
model.vol.grad
a.grad
model.vol.grad
model.initialState.grad
model.initialValue.grad

print(model.initialValue.grad)
print(process.discreteProcess[0:].grad)
model.vol.grad

a = torch.Tensor([2.]).requires_grad_()
a1 = torch.Tensor([3.]).requires_grad_()
b=torch.log(a)
d = b
e = d+1
c=e*a1
c.backward()

a.grad
a1.grad
d.grad
b.grad

import torch
a1 = torch.tensor([3.0,2.0,4.1])
b = a1[1].requires_grad_(True)
vol = torch.tensor(.2).requires_grad_(True)
a2 = torch.log(b)
b1 = vol*torch.tensor([0.1,0.2,-.2])
d = torch.exp(a2+torch.cumsum(b1,dim=0))[-1]
d.backward()
a1.grad
b.grad
vol.grad

#d = torch.mean(torch.exp(a2+torch.cumsum(b2,dim=0)))
#d = None
vol.grad.data.zero_()
a1.grad
a2.backward(c.grad)
a1.grad.data



### Example in scope
initialState=110.
discreteProcess = torch.empty((time.numberOfSteps+1))
#model = BlackScholesModel(0.2,0.03)
vol = 0.2
vol = torch.tensor(vol).requires_grad_(True)
riskFreeRate = torch.tensor(0.03)
drift = torch.sub(riskFreeRate,torch.mul(torch.mul(vol,vol),0.5))
factorLoadings = torch.tensor([vol])
initialState = torch.tensor(initialState).requires_grad_()
discreteProcess[0] = torch.log(initialState)

#drift = model.getDrift(0,0)
#factorLoadings = model.getFactorLoadings(0,0)
increments = driver.getIncrement(4)
a=torch.exp(torch.add(discreteProcess[0],torch.cumsum(torch.add(drift*time.deltaT,torch.mul(factorLoadings,increments)),dim=0)))
product.payoff(a[-1]).backward()
initialState.grad
model.vol.grad
