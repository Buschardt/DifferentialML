# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:29:53 2022

@author: magnu
"""
import numpy as np
import torch
class BrownianMotion():
    """
    Class defining a standard Brownian Motion
    Inputs:
        
        TimeDiscretization, Object - see TimeDiscretization
        numberOfPaths, # paths to generate
        numberOfFactors, # factors 
    
    
    """
    def __init__(self,TimeDiscretization,numberOfPaths,numberOfFactors,seed=None):
        self.TimeDiscretization = TimeDiscretization
        self.numberOfPaths = numberOfPaths
        self.numberOfFactors = numberOfFactors
        self.seed = seed
        self.increments = np.empty(0)
        
    def generateBM(self):
        if self.seed: np.random.seed(self.seed)
        self.increments = torch.normal(0,np.sqrt(self.TimeDiscretization.deltaT),(self.TimeDiscretization.getNumberOfSteps(),self.numberOfPaths))
    
    def getIncrement(self,pathNumber = None):
        if pathNumber == None:
            try:
                return self.increments
            except:
                print("generating proces")
                self.generateBM()
        else:
            try:
                return self.increments[:,pathNumber]
            except:
                print("generating proces")
                self.generateBM()
        
        
    
