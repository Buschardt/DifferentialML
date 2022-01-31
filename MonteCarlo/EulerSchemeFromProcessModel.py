# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:23:57 2022

@author: magnu
"""

import numpy as np


class EulerSchemeFromProcessModel:
    def __init__(self,model,stochasticDriver,timeDiscretization):
        self.model = model
        self.stochasticDriver = stochasticDriver
        self.timeDiscretization = timeDiscretization
        self.numberOfPaths = stochasticDriver.numberOfPaths
        self.discreteProcess = np.empty((self.timeDiscretization.numberOfSteps+1,self.numberOfPaths))
        
    def calculateProcess(self):
        self.discreteProcess[0,:] = self.model.initialState
        for i in range(self.timeDiscretization.numberOfSteps):
            drift = self.model.getDrift(i,self.discreteProcess[i-1])
            factorLoadings = self.model.getFactorLoadings(i,self.discreteProcess[i-1])
            increments = self.stochasticDriver.getIncrement(i)
            self.discreteProcess[i+1,:] = self.discreteProcess[i,:]+self.discreteProcess[i,:]*drift*self.timeDiscretization.deltaT
            self.discreteProcess[i+1,:] = self.discreteProcess[i,:]+self.discreteProcess[i,:]*factorLoadings*self.timeDiscretization.deltaT
        