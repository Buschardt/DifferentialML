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
        
    def payoff(self,terminalValue,riskFreeRate,timeToMaturity):
        if self.optType.lower() == 'call':
            return torch.exp(-riskFreeRate*timeToMaturity)*torch.maximum(terminalValue-self.strike,self.maxRHS)
        elif self.optType.lower() == 'put':
            return torch.exp(-riskFreeRate*timeToMaturity)*torch.maximum(self.strike-terminalValue,self.maxRHS)
        else:
            raise ValueError("Option type not recognized")
            
        