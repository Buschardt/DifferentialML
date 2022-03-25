# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:31:05 2022

@author: magnu
"""

import torch
import numpy as np
from scipy.stats import norm

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
            if _ =='vol':
                self.setDrift()
                self.setFactorLoadings()
            if _ == 'riskFreeRate':
                self.setDrift()
        
    def setDerivParameters(self,derivList=[]):
        self.derivParameters = derivList
        

    def getDerivParameters(self):
        return [getattr(self, _) for _ in self.derivParameters]
        
    def getParameters(self):
        return {'vol' : self.vol,
                'riskFreeRate': self.riskFreeRate}

    def setDrift(self):
        self.drift = self.riskFreeRate - .5*self.vol*self.vol
        
    def setFactorLoadings(self):
        self.factorLoadings = self.vol
        

#Analytics
def delta(S0, K, T, sigma, r, type='call'):
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r+(sigma**2)/2) * T )
    if type.lower() == 'call':
        return norm.cdf(d1)
    elif type.lower() == 'put':
        return -norm.cdf(-d1)

def vega(S0, K, T, sigma, r, type='call'):
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r+(sigma**2)/2) * T )
    if type.lower() == 'call' or type.lower() == 'put':
        return S0 * norm.pdf(d1) * np.sqrt(T)

def theta(S0, K, T, sigma, r, type='call'):
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r+(sigma**2)/2) * T )
    d2 = d1 - sigma * np.sqrt(T)
    
    if type.lower() == 'call':
        theta = -((S0 * norm.pdf(d1) * sigma)/(2*np.sqrt(T))) - r * K * np.exp(-r*T) * norm.cdf(d2)
        return theta
    elif type.lower() == 'put':
        theta = -((S0 * norm.pdf(d1) * sigma)/(2*np.sqrt(T))) + r * K * np.exp(-r*T) * norm.cdf(-d2)
        return theta

def rho(S0, K, T, sigma, r, type='call'):
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r+(sigma**2)/2) * T )
    d2 = d1 - sigma * np.sqrt(T)
    
    if type.lower() == 'call':
        rho = K * T * np.exp(-r*T) * norm.cdf(d2)
        return rho
    elif type.lower() == 'put':
        rho = -K * T * np.exp(-r*T) * norm.cdf(-d2)
        return rho
    
    
def gamma(S0, K, T, sigma, r, type='call'):
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r+(sigma**2)/2) * T )
    return norm.pdf(d1)/(S0*sigma*np.sqrt(T))

def V(S0, K, T, sigma, r, type='call'):
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r+(sigma**2)/2) * T )
    d2 = d1 - sigma * np.sqrt(T)

    if type.lower() == 'call':
        V = norm.cdf(d1)*S0 - norm.cdf(d2)*K*np.exp(-r*T)
    elif type.lower() == 'put':
        V = norm.cdf(-d2)*K*np.exp(-r*T) - norm.cdf(-d1)*S0
    return V
