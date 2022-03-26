# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:23:57 2022

@author: magnu
"""

#import torch
#from torch.autograd import Variable
import torch.autograd

class EulerSchemeFromProcessModel:
    def __init__(self,model,stochasticDriver,timeDiscretization,product):
        self.model = model
        self.stochasticDriver = stochasticDriver
        self.timeDiscretization = timeDiscretization
        self.discreteProcess = torch.empty((self.timeDiscretization.numberOfSteps+1))
        self.product = product
        self.checkCompatability()

    def calculateProcess(self,initialState,pathNumber,anti=False):
        self.initialState = initialState
        if self.model.stationary:
            drift = self.model.getDrift(0,0)
            factorLoadings = self.model.getFactorLoadings(0,0)
            increments = self.stochasticDriver.getIncrement(pathNumber)*-1 if anti else self.stochasticDriver.getIncrement(pathNumber)
            return torch.exp(torch.log(self.initialState) + torch.cumsum(drift * self.timeDiscretization.deltaT + factorLoadings * increments, axis=0))
        else:
            raise NotImplementedError()

    def pathPayoff(self, initialState, pathNumber,anti=False):
        S = self.calculateProcess(initialState, pathNumber)
        V = self.product.payoff(S,
                                self.model.riskFreeRate,
                                self.timeDiscretization.deltaT*self.timeDiscretization.numberOfSteps)
        if anti:
            Sa = self.calculateProcess(initialState, pathNumber,anti=True)
            Va = self.product.payoff(Sa,
                                    self.model.riskFreeRate,
                                    self.timeDiscretization.deltaT*self.timeDiscretization.numberOfSteps)
            return (S+Sa)/2,(V+Va)/2
        return S,V
    
    def calculateDerivs(self, initialState, pathNumber,second = False,anti = False):
        initialState = torch.tensor(initialState).requires_grad_()
        self.model.setAutoGradParams()
        self.model.setDrift()
        s,f = self.pathPayoff(initialState, pathNumber,anti)
        grad = torch.autograd.grad(f, [initialState]+self.model.getDerivParameters(), allow_unused=True, create_graph=second)
        if second:
            secgrads = torch.autograd.grad(grad[0],[initialState])[0]
            return s,f,grad, secgrads
        return s,f,grad
    
    def checkCompatability(self):
        if self.model.getNumberOfFactors() != self.stochasticDriver.numberOfFactors:
            raise ValueError("Number of factors in model and stochatic Driver must be equal")
        
    
