# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:31:05 2022

@author: magnu
"""
import numpy as np
class BlackScholesModel:
    def __init__(self,initialValue,vol,riskFreeRate):
        self.initialValue = initialValue
        self.vol = vol
        self.riskFreeRate = riskFreeRate
        self.initialState = np.log(self.initialValue)
        self.drift = self.riskFreeRate - self.vol*self.vol/2
        self.factorLoadings = vol