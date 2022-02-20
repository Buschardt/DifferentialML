# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:23:57 2022

@author: magnu
"""

#import torch
from torch.autograd import Variable
import torch.autograd

class EulerSchemeFromProcessModel:
    def __init__(self,model,stochasticDriver,timeDiscretization,product):
        self.model = model
        self.stochasticDriver = stochasticDriver
        self.timeDiscretization = timeDiscretization
        self.discreteProcess = torch.empty((self.timeDiscretization.numberOfSteps+1))
        self.product = product

    def calculateProcess(self,initialState,pathNumber):
        self.initialState = initialState
        #self.discreteProcess[0] = torch.log(self.initialState)
        if self.model.stationary:
            drift = self.model.getDrift(0,0)
            factorLoadings = self.model.getFactorLoadings(0,0)
            increments = self.stochasticDriver.getIncrement(pathNumber)
            return torch.exp(torch.log(self.initialState) + torch.cumsum(drift * self.timeDiscretization.deltaT + factorLoadings * increments, axis=0))
            #return torch.exp(torch.add(self.discreteProcess[0],torch.cumsum(torch.add(drift*self.timeDiscretization.deltaT,torch.mul(factorLoadings,increments)),dim=0)))
            #self.discreteProcess[1:] = torch.add(self.initialState,torch.cumsum(torch.add(drift*self.timeDiscretization.deltaT,torch.mul(factorLoadings,increments)),dim=0))    
            #return torch.exp(self.discreteProcess)
        else:
            raise NotImplementedError()

    def pathPayoff(self, initialState, pathNumber):
        S = self.calculateProcess(initialState, pathNumber)
        V = self.product.payoff(S)

        return V
    
    def calculateDerivs(self, initialState, pathNumber):
        initialState = torch.tensor(initialState).requires_grad_()
        f = self.pathPayoff(initialState, pathNumber)[-1]
        grad = torch.autograd.grad(f, [initialState], allow_unused=True)
            
        return grad[0]
