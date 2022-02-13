# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:31:05 2022

@author: magnu
"""
import numpy as np

from .AbstractProcessClass import AbstractProcessClass

class BlackScholesModel(AbstractProcessClass):
    def __init__(self,initialValue,vol,riskFreeRate):
        self.initialValue = initialValue
        self.vol = vol
        self.riskFreeRate = riskFreeRate
        self.initialState = np.log(self.initialValue)
        self.drift = self.riskFreeRate - self.vol*self.vol/2
        self.factorLoadings = np.array([vol])
        
    def getDrift(self,timeindex,state):
        return self.drift
    def getFactorLoadings(self,timeindex,state):
        return self.factorLoadings
    
    def getNumberOfFactors(self):
        return 1
    
