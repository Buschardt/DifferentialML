# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:31:05 2022

@author: magnu
"""

from torch.autograd import Variable
import torch

from .AbstractProcessClass import AbstractProcessClass

class BlackScholesModel(AbstractProcessClass):
    def __init__(self,vol,riskFreeRate):
        self.vol = torch.tensor(vol).requires_grad_(True)
        self.riskFreeRate = torch.tensor(riskFreeRate)
        self.drift = torch.sub(self.riskFreeRate,torch.mul(torch.mul(self.vol,self.vol),0.5))
        self.factorLoadings = torch.tensor([vol])
        self.stationary = True
        
    def getDrift(self,timeindex,state):
        return self.drift
    def getFactorLoadings(self,timeindex,state):
        return self.factorLoadings
    
    def getNumberOfFactors(self):
        return 1
    def getParameters(self):
        return {'vol' : self.vol,
                'riskFreeRate': self.riskFreeRate}
    
