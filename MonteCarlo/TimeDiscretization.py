# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:29:31 2022

@author: magnu
"""

class TimeDiscretization:
    def __init__(self,initial,numberOfSteps,deltaT):
        self.initial = initial
        self.numberOfSteps = numberOfSteps
        self.deltaT = deltaT
    
    def getTimeMap(self):
        return [i*self.deltaT for i in range(self.initial,self.initial+self.numberOfSteps)]
    
    def getNumberOfSteps(self):
        return self.numberOfSteps