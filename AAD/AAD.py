# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 21:26:06 2022

@author: magnu
"""

#AAD implementation
from collections import defaultdict
import numpy as np
from scipy.stats import norm

def sin(a):
    value = np.sin(a.value)
    local_gradients = (
        (a, np.cos(a.value)),
    )
    return Variable(value, local_gradients)

def sqrt(a):
    value = np.sqrt(a.value)
    local_gradients = (
        (a,0.5*pow(a.value,-0.5)),
        )
    return Variable(value,local_gradients)

def normcdf(a):
    value = norm.cdf(a.value)
    local_gradients = (
        (a,norm.pdf(a.value)),
        )
    return Variable(value,local_gradients)

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
            print('path_value: ',path_value)
            # "Multiply the edges of a path":
            value_of_path_to_child = path_value * local_gradient
            print('value_of_path_to_child: ',value_of_path_to_child)
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
_05 = Variable(0.5)
r=Variable(0.03)
T=Variable(2)
S0=Variable(40)
K=Variable(40)
sigma=Variable(0.2)
dfVar = exp(neg(r)*T)
forwardVar = S0*exp(r*T)
get_gradients(forwardVar)
stdVar = sigma*sqrt(T)
moneyness = log(forwardVar/K)/stdVar
d1 = moneyness+stdVar*_05
d2 = moneyness-stdVar*_05
P = dfVar*(K*normcdf(neg(d2))-forwardVar*normcdf(neg(d1)))
gradients = get_gradients(P)
gradients[S0]
gradients[sigma]
gradients[r]
gradients[T]
get_gradients(exp(neg(r)))
a = Variable(0.1)
r = a*b
b = Variable(0.3)
c = a* b+a
gradients = get_gradients(c)
for i in gradients:
    print(i.value)
    print(i)
c.localGradients
b.localGradients
a.localGradients

s = Variable(100)
factor = Variable(np.exp(1.02))
s = s*factor
get_gradients(s)
s.value
s.localGradients

