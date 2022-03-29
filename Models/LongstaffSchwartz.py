from dis import disco
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy.polynomial.polynomial as poly
import numpy as np
import matplotlib.pyplot as plt

#Linear model class
class LinearModel(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)

    def forward(self, x):
        return self.linear(x)

    def train(self, X, y, n_epochs=100, batch_size=256):

        loss = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr = 0.001)
        for epoch in range(n_epochs):

            permutation = torch.randperm(X.size()[0])

            for i in range(0, X.size()[0], batch_size):
                indicies = permutation[i:i+batch_size]
                X_batch, y_batch = X[indicies], y[indicies]
                self.zero_grad()
                output = self(X_batch)
                loss = F.mse_loss(output, y_batch)
                loss.backward()
                optimizer.step()

#Creates feature matrix from state variable
def featureMatrix(state):
    state = state.view(state.shape[0], 1)
    x1 = state
    x2 = torch.pow(state, 2)
    x3 = torch.pow(state, 3)
    x4 = torch.pow(state, 4)
    X = torch.cat((x1, x2, x3, x4), dim = 1)
    return X



def predExerciseBoundary(xTrain,yTrain,xTest,degree = 2,basis = 'polynomial'):
    if basis == 'polynomial':
        coefs = poly.polyfit(xTrain, yTrain, degree,rcond=None)
    elif basis == 'laguerre':
        coefs = np.polynomial.laguerre.lagfit(xTrain,yTrain,degree,rcond=None)
        return np.polynomial.laguerre.lagval(xTest,coefs).T
    elif basis == 'legendre':
        coefs = np.polynomial.legendre.legfit(xTrain,yTrain,degree,rcond=None)
        return np.polynomial.legendre.legval(xTest,coefs).T
    return poly.polyval(xTest,coefs).T



#Simulates Black-Scholes paths and calculates exercise value at each t
def genPaths(S, K, sigma, r, T, dt, dW, type='call', anti=False,tp = None):
    if len(dW.shape) == 2:
        axis = 1
    else:
        axis = 0
    if tp:
        dts = torch.concat((torch.tensor([0.]),torch.tensor([T/(dt)]*dt)))[:tp]
    else:
        dts = torch.concat((torch.tensor([0.]),torch.tensor([T/(dt)]*(dt))))
    St = S * torch.cumprod(torch.exp((r-sigma**2/2)*dts + sigma*torch.sqrt(dts)*dW), axis=axis)
    if type == 'call':
        Et = torch.maximum(St-K, torch.tensor(0))
    elif type == 'put':
        Et = torch.maximum(K-St, torch.tensor(0))
    if anti == True:
        dW_anti = -1 * dW
        St_anti = S * torch.cumprod(torch.exp((r-sigma**2/2)*dts + sigma*torch.sqrt(dts)*dW_anti), axis=axis)
        if type == 'call':
            Et_anti = torch.maximum(St_anti-K, torch.tensor(0))
        elif type == 'put':
            Et_anti = torch.maximum(K-St_anti, torch.tensor(0))
        return torch.cat((St, St_anti), 0), torch.cat((Et, Et_anti), 0)
    return St, Et

#Trains Linear models for each timepoint t used to estimate the continuation values.
#Starts at time T, and iterates backwards through time.
#Returns model parameters in list
def LSM_train(St, Et):
    modelw = [] #list for weights
    modelb = [] #list for biases
    n_excerises = St.shape[1]

    St = St.float()
    Et = Et.float()

    for i in range(n_excerises-1)[::-1]:
        y = Et[:, i+1:i+2]
        X = St[:, i]
        X = featureMatrix(X)
        tempModel = LinearModel(4)
        tempModel.train(X[Et[:,i]>0], y[Et[:,i]>0]) #trains on paths ITM
        continuationValue = tempModel(X) 
        Et[:, i:i+1] = torch.maximum(Et[:, i:i+1], continuationValue).detach()

        modelw.insert(0, tempModel.linear.weight)
        modelb.insert(0, tempModel.linear.bias)
        del continuationValue
    
    return modelw, modelb

def putPayoff(st,k):
    k = np.array([k])
    return np.maximum(k-st,0)

def LSM_train_poly(St, Et,discount):
    n_excerises = St.shape[1]
    Tt = np.array([n_excerises]*Et.shape[0])
    St = St.numpy()
    
    Et = Et.numpy()
    Tt = np.where(Et[:, -1]>0,Tt,n_excerises)
    cashflow = Et[:,-1]
    for i in range(n_excerises-1)[::-1]:
        cashflow = cashflow*discount
        X = St[:, i]
        exercise = Et[:,i]
        itm = exercise>0
        try:
            continuationValue = predExerciseBoundary(X[itm], cashflow[itm], X,3,basis = 'laguerre')       
            ex_idx = (exercise>continuationValue)*itm
            Tt = np.where(ex_idx,i,Tt)
            cashflow[ex_idx] = exercise[ex_idx]
            del continuationValue
        except:
            pass
        
    
    return Tt,cashflow


def simpleLSM(S,K,sigma,r,T,dt,dW,type = 'call', anti = False):
    if anti:
        St, Et = genPaths(S, K, sigma, r, T, dt, dW[0], type=type, anti=False,tp = dW[0].nelement())
        St_anti, Et_anti = genPaths(S, K, sigma, r, T, dt, dW[1], type=type, anti=False,tp = dW[1].nelement())
        V = (Et[-1]*np.exp(-r*T*(len(dW[0])-1)/dt) + Et_anti[-1]*np.exp(-r*T*(len(dW[1])-1)/dt)) / 2
        return V
    #if not dW.nelement():
    #    return torch.maximum(K-S,torch.tensor(0)) if type == 'put' else torch.maximum(S-K,torch.tensor(0))
    St, Et = genPaths(S, K, sigma, r, T, dt, dW, type=type, anti=False,tp = dW.nelement())
    return Et[-1]*np.exp(-r*T*(len(dW)-1)/dt)
    

#Longstaff-Schwartz algorithm using estimated models from LSM_train
def LSM(S, K, sigma, r, T, dt, dW, w, b, type='call', anti=False):
    St, Et = genPaths(S, K, sigma, r, T, dt, dW, type=type, anti=anti)
    discountFactor = np.exp(-r * (T/dt))
    continuationValues = []
    exercises = []
    previous_exercises = 0
    npv = 0

    for i in range(dt-1):
        X = featureMatrix(St[:, i]).double()
        contValue = torch.add(torch.matmul(X.double(), w[i].T.double()), b[i].double())
        continuationValues.append(contValue)
        inMoney = torch.greater(Et[:,i], 0.).float()
        exercise = torch.greater(Et[:,i], contValue[:,0]).float() * inMoney 
        exercises.append(exercise)
        exercise = exercise * (1-previous_exercises)
        previous_exercises += exercise
        npv += exercise*Et[:,i] * discountFactor
    
    #Last exercise date
    inMoney = torch.greater(Et[:,-1], 0.).float()
    exercise =  inMoney * (1-previous_exercises)
    npv += exercise*Et[:,-1] * discountFactor
    npv = torch.mean(npv)

    return npv


def standardLSM(S0, K,sigma,r,T,dt,discount,nPaths,dW = None):
    if dW is None:
        dW = torch.randn(nPaths, dt + 1)
    n_excerises = dt + 1
    dts = torch.concat((torch.tensor([0.]),torch.tensor([T/(dt)]*(dt))))
    S0 = torch.tensor([(S0,)]*nPaths)
    St = S0 * torch.cumprod(torch.exp((r-sigma**2/2)*dts + sigma*torch.sqrt(dts)*dW), axis=1)
    Et = np.maximum(K-St,0)
    cashflow = Et[:,-1]
    for i in range(1,n_excerises-1)[::-1]:
        cashflow = cashflow*discount
        X = St[:, i]
        exercise = Et[:,i]
        itm = exercise>0
        try:
            continuationValue = predExerciseBoundary(X[itm], cashflow[itm], X,2,basis = 'polynomial')
            ex_idx = (exercise>continuationValue)*itm
            cashflow[ex_idx] = exercise[ex_idx]
        except:
            pass
    return (cashflow * discount).mean()

def finiteDifferenceLSM(S0, K,sigma,r,T,dt,discount,nPaths,eps=0.001):
    dW = torch.randn(nPaths, dt + 1)
    return (standardLSM(S0+eps,K,sigma,r,T,dt,discount,nPaths,dW)-standardLSM(S0-eps,K,sigma,r,T,dt,discount,nPaths,dW))/(2*eps)

