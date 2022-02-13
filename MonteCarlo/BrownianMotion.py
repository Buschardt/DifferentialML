# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:29:53 2022

@author: magnu
"""
import numpy as np
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
        self.increments = np.random.normal(0,np.sqrt(self.TimeDiscretization.deltaT),(self.TimeDiscretization.getNumberOfSteps(),self.numberOfFactors,self.numberOfPaths))
    
    def getIncrement(self,timeIndex):
        try:
            return self.increments[timeIndex,:,:]
        except:
            print("generating proces")
            self.generateBM()
        