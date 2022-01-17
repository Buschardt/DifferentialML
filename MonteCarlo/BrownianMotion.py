# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:29:53 2022

@author: magnu
"""
import numpy as np
class BrownianMotion():
    
    def __init__(self,TimeDiscretization,numberOfPaths,numberOfFactors,seed):
        self.TimeDiscretization = TimeDiscretization
        self.numberOfPaths = numberOfPaths
        self.numberOfFactors = numberOfFactors
        self.seed = seed
        
    def generateBM(self):
        np.random.seed(self.seed)
        BMincrements = np.random.normal(0,np.sqrt(self.TimeDiscretization.deltaT),(self.TimeDiscretization.getNumberOfSteps(),self.numberOfFactors,self.numberOfPaths))
        return BMincrements