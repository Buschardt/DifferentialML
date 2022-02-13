#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import PytorchAAD as aad
import numpy as np
#%%
class NeuralNet(nn.Module):
    def __init__(self, dimInput, dimOutput, nLayers, nHiddenNeurons):
        super().__init__()

        self.layers = nn.Sequential()

        self.layers.add_module(f'fc{0}', nn.Linear(dimInput, nHiddenNeurons)) #Input layer
        self.layers.add_module(f'activation {0}', nn.ELU())
        for i in range(1, nLayers - 1):
            self.layers.add_module(f'fc{i}', nn.Linear(nHiddenNeurons, nHiddenNeurons)) #Hidden layers
            self.layers.add_module(f'activation {i}', nn.ELU())
        self.layers.add_module(f'fc{nLayers - 1}', nn.Linear(nHiddenNeurons, dimOutput)) #output layer
        self.layers.add_module(f'activation {nLayers - 1}', nn.Softplus())

    def forward(self, x):
        tensor = self.layers(x)
        return tensor

def trainingLoop(X, y, n_epochs, batch_size, NeuralNet):
    optimizer = optim.Adam(NeuralNet.parameters(), lr=0.001)

    for epoch in range(n_epochs):

        permutation = torch.randperm(X.size()[0])

        for i in range(0,X.size()[0], batch_size):
            indicies = permutation[i:i+batch_size]
            X_batch, y_batch = X[indicies], y[indicies]
            NeuralNet.zero_grad()
            output = NeuralNet(X_batch)
            loss = F.mse_loss(output, y_batch)
            loss.backward()
            optimizer.step()
        print(loss)

#Get gradients of NN w.r.t. inputs
def backprop(X, net):
    gradients = torch.autograd.grad(net(X), X, grad_outputs=torch.ones(X.shape[0], X.shape[1]), create_graph=True, retain_graph=True, allow_unused=True)
    return gradients 


#define custom loss function with differential regularization
def diffReg(labels, predictions, derivLabels, derivPredictions, alpha, beta):
    loss_values = alpha * F.mse_loss(labels, predictions)
    loss_derivs = beta * F.mse_loss(derivLabels, derivPredictions)
    return loss_values + loss_derivs

#Training loop with differential regularization
def diffTrainingLoop(X, y_values, y_derivs, n_epochs, batch_size, NeuralNet, alpha, beta):
    optimizer = optim.Adam(NeuralNet.parameters(), lr=0.001)

    for epoch in range(n_epochs):

        permutation = torch.randperm(X.size()[0])

        for i in range(0,X.size()[0], batch_size):
            indicies = permutation[i:i+batch_size]
            X_batch, y_values_batch, y_derivs_batch = X[indicies], y_values[indicies], y_derivs[indicies]
            NeuralNet.zero_grad()
            output = NeuralNet(X_batch)
            X_batch.requires_grad_()
            predDerivs = backprop(X_batch, NeuralNet) #compute derivatives of net
            loss = diffReg(y_values_batch, output, y_derivs_batch, predDerivs[0], alpha, beta)
            loss.backward()
            optimizer.step()
        print(loss)

#Preprocessing
def standardscale(X, y, dydx = None):
    X_mean = X.mean(0, keepdim = True)
    X_std = X.std(0, unbiased=False, keepdim=True)
    X = (X - X_mean) / X_std

    y_mean = y.mean(0, keepdim = True)
    y_std = y.std(0, unbiased=False, keepdim=True)
    y = (y - y_mean) / y_std

    #scale derivatives
    if dydx is not None:
        dydx = dydx / y_std * X_std
        lambda_j = 1.0 / torch.sqrt((dydx ** 2).mean(0)).reshape(1, -1)
    else:
        lambda_j = None

    return X, y, dydx, lambda_j, X_mean, X_std, y_mean, y_std

def unscale(X, y, dydx, X_mean, X_std, y_mean, y_std):
    X = X_mean + X_std * X
    y = y_mean + y_std * y

    dydx = y_std / X_std * dydx

    return X, y, dydx

#%%
###BS example - Standard ML###
#Define variables
nSamples = 10000
S0_0 = 0
S0_n = 2.25
S0 = torch.linspace(S0_0, S0_n, nSamples)
K = torch.tensor([1.])
T = torch.tensor([1.])
sigma = torch.tensor([0.2])
r = torch.tensor([0.03])

#Define NN
net = NeuralNet(1, 1, 6, 20)
#Simulate and derivatives
y = aad.euroCallPrice(S0, K, T, sigma, r, 100, 10, 1)
y_derivs = torch.tensor(aad.deltaBS(S0, K, T, sigma, r))

#shaped for NN
S0 = S0.view(nSamples,1).float()
y = y.view(nSamples,1).float()
y_derivs = y_derivs.view(nSamples,1).float()

#Train
trainingLoop(S0, y, 3, 10, net)

#Generate approximations from trained NN
X_test = torch.linspace(S0_0, S0_n, 100).view(100,1)
y_test = net(X_test).detach().numpy()

X_test.requires_grad_()
y_derivs_test = backprop(X_test, net)[0].detach().numpy()
X_test = X_test.detach().numpy()

#plot results
plt.figure(figsize=[14,8])
plt.subplot(1, 2, 1)
plt.plot(S0, y, 'o', label='Simulated prices')
plt.plot(X_test, y_test, color='black', label='NN approximation')
plt.xlabel('S0')
plt.ylabel('C')
plt.legend()
plt.title('Standard ML - Price approximation')

plt.subplot(1, 2, 2)
plt.plot(S0, y_derivs, 'o', label='True BS delta')
plt.plot(X_test, y_derivs_test, color='black', label='NN approximated deltas')
plt.xlabel('S0')
plt.ylabel('Delta')
plt.legend()
plt.title('Standard ML - Delta approximation')
plt.savefig('StandardML.png')

#%%
###BS example - Differential ML###
#Define NN
diffNet = NeuralNet(1, 1, 6, 20)

#Train
diffTrainingLoop(S0, y, y_derivs, 3, 10, diffNet, 0.25, 0.75)

#Generate approximations from trained NN
X_test = torch.linspace(S0_0, S0_n, 100).view(100,1)
y_test = diffNet(X_test).detach().numpy()

X_test.requires_grad_()
y_derivs_test = backprop(X_test, diffNet)[0].detach().numpy()
X_test = X_test.detach().numpy()

#plot results
plt.figure(figsize=[14,8])
plt.subplot(1, 2, 1)
plt.plot(S0, y, 'o', label='Simulated prices')
plt.plot(X_test, y_test, color='black', label='NN approximation')
plt.xlabel('S0')
plt.ylabel('C')
plt.legend()
plt.title('Differential ML - price approximation')

plt.subplot(1, 2, 2)
plt.plot(S0, y_derivs, 'o', label='True BS delta')
plt.plot(X_test, y_derivs_test, color='black', label='NN approximated deltas')
plt.xlabel('S0')
plt.ylabel('Delta')
plt.legend()
plt.title('Differential ML - Delta approximation')
plt.savefig('DiffML.png')

#%%

