# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 22:42:35 2022

@author: magnu
"""
import numpy as np
def genCorrel(num_stocks):
    randoms = np.random.uniform(low=-1., high=1., size=(2*num_stocks, num_stocks))
    cov = randoms.T @ randoms
    invvols = np.diag(1. / np.sqrt(np.diagonal(cov)))
    return np.linalg.multi_dot([invvols, cov, invvols])
