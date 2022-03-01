# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:38:14 2022

@author: magnu
"""

import torch

class Option:
    def __init__(self, strike, optType='call'):
        self.strike = strike
        self.optType = optType
        self.maxRHS = torch.tensor(0.)
        
    def payoff(self,path,riskFreeRate,timeToMaturity):
        if self.optType.lower() == 'call':
            return torch.exp(-riskFreeRate*timeToMaturity)*torch.maximum(path[-1]-self.strike,self.maxRHS)
        elif self.optType.lower() == 'put':
            return torch.exp(-riskFreeRate*timeToMaturity)*torch.maximum(self.strike-path[-1],self.maxRHS)
        else:
            raise ValueError("Option type not recognized")
            
            
class BasketOption:
    def __init__(self,strike,numberOfFactors,weights,optType = 'call'):
        self.strike = strike
        self.optType = optType
        self.weights = torch.tensor(weights)
        self.numberOfFactors = numberOfFactors
        self.maxRHS = torch.tensor([0.])
        
    def payoff(self,path,riskFreeRate,timeToMaturity):
        if self.optType.lower() == 'call':
            return torch.exp(-riskFreeRate*timeToMaturity)*torch.maximum(torch.sum(self.weights*path[-1,:])-self.strike,self.maxRHS)
        elif self.optType.lower() == 'put':
            return torch.exp(-riskFreeRate*timeToMaturity)*torch.maximum(self.strike-torch.sum(self.weights*path[-1,:]),self.maxRHS)
        else:
            raise ValueError("Option type not recognized")