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
        
    def calculateProcess(self,initialState,pathNumber):
        self.initialState = torch.tensor(initialState).requires_grad_()
        self.discreteProcess[0] = torch.log(self.initialState)
        if self.model.stationary:
            drift = self.model.getDrift(0,0)
            factorLoadings = self.model.getFactorLoadings(0,0)
            increments = self.stochasticDriver.getIncrement(pathNumber)
            return torch.exp(torch.add(self.discreteProcess[0],torch.cumsum(torch.add(drift*self.timeDiscretization.deltaT,torch.mul(factorLoadings,increments)),dim=0)))
            #self.discreteProcess[1:] = torch.add(self.initialState,torch.cumsum(torch.add(drift*self.timeDiscretization.deltaT,torch.mul(factorLoadings,increments)),dim=0))    
            #return torch.exp(self.discreteProcess)
        else:
            raise NotImplementedError()
            
        
