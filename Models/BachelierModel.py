# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:31:27 2022

@author: magnu
"""

import torch

from AbstractProcessClass import AbstractProcessClass


class BachelierModel(AbstractProcessClass):
    def __init__(self,riskFreeRate,numSecs,volVec,corMatrix):
        raise NotImplementedError()
        self.volVec = torch.tensor(volVec)
        self.corMatrix = torch.tensor(corMatrix)
        self.riskFreeRate = torch.tensor(riskFreeRate)
        self.drift = self.riskFreeRate
        self.NumberOfFactors = numSecs
        self.factorLoadings = torch.tensor(self.volVec)
        self.stationary = True
        self.derivParameters = []
        
        

