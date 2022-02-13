# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:04:54 2022

@author: magnu
"""
import os
os.chdir(r'C:\Users\magnu\DifferentialML')
from Models.BlackScholesModel import BlackScholesModel
from MonteCarlo.TimeDiscretization import TimeDiscretization
from MonteCarlo.BrownianMotion import BrownianMotion
from MonteCarlo.EulerSchemeFromProcessModel import EulerSchemeFromProcessModel

time = TimeDiscretization(0,100,0.01)
driver = BrownianMotion(time,10000,1)
driver.generateBM()
model = BlackScholesModel(100,0.2,0.03)
process = EulerSchemeFromProcessModel(model,driver,time)
process.calculateProcess()
a = process.discreteProcess

