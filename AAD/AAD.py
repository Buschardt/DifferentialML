# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 21:26:06 2022

@author: magnu
"""

#AAD implementation
from collections import defaultdict
import numpy as np

def sin(a):
    value = np.sin(a.value)
    local_gradients = (
        (a, np.cos(a.value)),
    )
    return Variable(value, local_gradients)

def exp(a):
    value = np.exp(a.value)
    local_gradients = (
        (a, value),
    )
    return Variable(value, local_gradients)
    
def log(a):
    value = np.log(a.value)
    local_gradients = (
        (a, 1. / a.value),
    )
    return Variable(value, local_gradients)

def add(a, b):
    "Create the variable that results from adding two variables."
    value = a.value + b.value    
    local_gradients = (
        (a, 1),  # the local derivative with respect to a is 1
        (b, 1)   # the local derivative with respect to b is 1
    )
    return Variable(value, local_gradients)

def mul(a, b):
    "Create the variable that results from multiplying two variables."
    value = a.value * b.value
    local_gradients = (
        (a, b.value), # the local derivative with respect to a is b.value
        (b, a.value)  # the local derivative with respect to b is a.value
    )
    return Variable(value, local_gradients)

def neg(a):
    value = -1 * a.value
    local_gradients = (
        (a, -1),
    )
    return Variable(value, local_gradients)

def inv(a):
    value = 1. / a.value
    local_gradients = (
        (a, -1 / a.value**2),
    )
    return Variable(value, local_gradients)   


class Variable:
    def __init__(self,value,localGradients = ()):
        self.value = value
        self.localGradients = localGradients
        
    def __add__(self,other):
        return add(self,other)
    
    def __mul__(self,other):
        return mul(self,other)
    
    def __sub__(self,other):
        return add(self,neg(other))
    
    def __truediv__(self, other):
        return mul(self, inv(other))
    
def get_gradients(variable):
    """ Compute the first derivatives of `variable` 
    with respect to child variables.
    """
    gradients = defaultdict(lambda: 0)
    
    def compute_gradients(variable, path_value):
        for child_variable, local_gradient in variable.localGradients:
            # "Multiply the edges of a path":
            value_of_path_to_child = path_value * local_gradient
            # "Add together the different paths":
            gradients[child_variable] += value_of_path_to_child
            # recurse through graph:
            compute_gradients(child_variable, value_of_path_to_child)
    
    compute_gradients(variable, path_value=1)
    # (path_value=1 is from `variable` differentiated w.r.t. itself)
    return gradients
    
a = Variable(1000)
b = Variable(2)
c = Variable(2)
a.localGradients
y = a*b
y.localGradients
def f(a, b, c):
    return a*a
y = f(a,b,c)
y.localGradients
m = get_gradients(y)
print(m[a])

test = Variable(y)
y.value
y.localGradients[1]
a = Variable(4)
b = Variable(3)
c = add(a, b) # = 4 + 3 = 7
d = mul(a, c) # = 4 * 7 = 28

a = Variable(4)
b = Variable(3)
c = a + b
d = a*c
gradients = get_gradients(d)
d.localGradients
c.localGradients
b.localGradients
a.localGradients

s = Variable(100)
factor = Variable(np.exp(1.02))
s = s*factor
get_gradients(s)
s.value
s.localGradients
