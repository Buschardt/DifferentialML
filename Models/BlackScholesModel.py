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
        self.vol = torch.tensor(vol)
        self.riskFreeRate = torch.tensor(riskFreeRate)
        self.drift = self.riskFreeRate - .5*self.vol*self.vol
        self.factorLoadings = torch.tensor([vol])
        self.stationary = True
        self.derivParameters = []
        
    def getDrift(self,timeindex,state):
        return self.drift
    def getFactorLoadings(self,timeindex,state):
        return self.vol
    def getNumberOfFactors(self):
        return 1
    
    def setAutoGradParams(self):
        for _ in self.derivParameters:
            setattr(self, _, torch.tensor(getattr(self, _)).requires_grad_())
        
    def setDerivParameters(self,derivList=[]):
        self.derivParameters = derivList
        
    def getDerivParameters(self):
        return [getattr(self, _) for _ in self.derivParameters]
        
    def getParameters(self):
        return {'vol' : self.vol,
                'riskFreeRate': self.riskFreeRate}
    