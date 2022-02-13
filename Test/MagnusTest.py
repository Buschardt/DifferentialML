# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 09:40:12 2022

@author: magnu
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm

x = torch.randn(5, requires_grad=True)
y = x.exp()
print(x.equal(y.grad_fn._saved_self))  # True
print(x is y.grad_fn._saved_self)  # True
x
y.grad_fn._saved_result
y.grad_fn._saved_self
y.grad_fn.saved_tensors
S0= 100
S0 = S0 *np.exp(np.random.normal(0, 1,1000))
plt.plot(S0)
plt.plot(1)
