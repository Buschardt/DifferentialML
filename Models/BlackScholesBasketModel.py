# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 22:07:15 2022

@author: magnu
"""

from torch.autograd import Variable
import torch

from AbstractProcessClass import AbstractProcessClass

class MultiAssetBlackScholes(AbstractProcessClass):
    def __init__(self,riskFreeRate,numSecs,volVec):
        self.volVec = torch.tensor(volVec)
        self.riskFreeRate = torch.tensor(riskFreeRate)
        self.drift = self.riskFreeRate - .5*self.volVec.T*self.volVec
        self.NumberOfFactors = numSecs
        self.factorLoadings = torch.tensor(self.volVec)
        self.stationary = True
        self.derivParameters = []
        
    def getDrift(self,timeindex,state):
        return self.drift
    def getFactorLoadings(self,timeindex,state):
        return self.factorLoadings
    def getNumberOfFactors(self):
        return self.NumberOfFactors
    
    def setAutoGradParams(self):
        for _ in self.derivParameters:
            setattr(self, _, torch.tensor(getattr(self, _)).requires_grad_())
            if _ =='volVec':
                self.setDrift()
                self.setFactorLoadings()
            if _ == 'riskFreeRate':
                self.setDrift()
        
    def setDerivParameters(self,derivList=[]):
        self.derivParameters = derivList
        

    def getDerivParameters(self):
        return [getattr(self, _) for _ in self.derivParameters]
        
    def getParameters(self):
        return {'volVec' : self.volVec,
                'riskFreeRate': self.riskFreeRate}
    
    def setDrift(self):
        self.drift = self.riskFreeRate - .5*self.volVec.T*self.volVec
        
    def setFactorLoadings(self):
        self.factorLoadings = torch.tensor(self.volVec)
        

