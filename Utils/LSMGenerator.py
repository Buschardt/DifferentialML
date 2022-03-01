# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 00:29:37 2022

@author: magnu
"""
import numpy as np
class LSMGenerator():
    def __init__(self,spots,diffuse,numberOfFactors):
        self.diffuse = diffuse
        self.numberOfFactors = numberOfFactors
        self.spots = spots
        
    def LSMspots(self,nSamples):
        return self.spots + self.diffuse * np.random.normal(0, 1, (nSamples,self.numberOfFactors))    