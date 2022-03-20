from dis import disco
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy.polynomial.polynomial as poly
import numpy as np

#Linear model class
class LinearModel(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)

    def forward(self, x):
        return self.linear(x)

    def train(self, X, y, n_epochs=100, batch_size=257):

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



def predExerciseBoundary(xTrain,yTrain,xTest,degree = 4):
    coefs = poly.polyfit(xTrain, yTrain, degree)
    return poly.polyval(xTest,coefs).T


#Simulates Black-Scholes paths and calculates exercise value at each t
def genPaths(S, K, sigma, r, T, dt, dW, type='call', anti=False,tp = None):
    
    if len(dW.shape) == 2:
        axis = 1
    else:
        axis = 0
    
    if tp:
        dts = torch.tensor([T/dt]*dt)[:tp]
    else:
        dts = torch.tensor([T/dt]*dt).view(1,dt)
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
        return (St+St_anti)/2, (Et+Et_anti)/2
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


def LSM_train_poly(St, Et):
    n_excerises = St.shape[1]
    Tt = np.array([n_excerises]*Et.shape[0])
    St = St.numpy()
    Et = Et.numpy()
    Tt = np.where(Et[:, -1]>0,Tt,n_excerises)
    for i in range(n_excerises-1)[::-1]:
        y = Et[:, i+1]
        X = St[:, i]
        continuationValue = predExerciseBoundary(X[Et[:,i]>0],  y[Et[:,i]>0], X)
        inMoney = np.greater(Et[:,i], 0.)
        Tt = np.where((Et[:, i]>continuationValue)*inMoney,i,Tt)
        Et[:, i] = np.maximum(Et[:, i], continuationValue)
        del continuationValue
    
    return Tt


def simpleLSM(S,K,sigma,r,T,dt,dW,type = 'call',anti = False):
    if not dW.nelement():
        return torch.maximum(K-S,torch.tensor(0)) if type == 'put' else torch.maximum(S-K,torch.tensor(0))
    
    St, Et = genPaths(S, K, sigma, r, T, dt, dW, type=type, anti=False,tp = dW.nelement())
    return Et[-1]*np.exp(-r*T*len(dW)/dt)
    

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