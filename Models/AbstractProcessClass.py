# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 20:40:03 2022

@author: magnu
"""


from abc import ABC, abstractmethod

class AbstractProcessClass(ABC):
    @abstractmethod
    def getDrift(self):
        pass
    @abstractmethod
    def getFactorLoadings(self):
        pass
    @abstractmethod
    def getNumberOfFactors(self):
        pass
    
    