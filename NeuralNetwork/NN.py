import torch
import torch.nn as nn
import NeuralNetwork.Training as Training
import NeuralNetwork.Preprocessing as pre

class NeuralNet(nn.Module):
    def __init__(self, dimInput, dimOutput, nHiddenLayers, nHiddenNeurons, differential=False):
        #Inherit from nn.Module
        super().__init__()
        self.differential = differential
        self.X_scaled = None

        #Define sequential NN
        self.layers = nn.Sequential()

        #Input layer
        self.layers.add_module(f'fc{0}', nn.Linear(dimInput, nHiddenNeurons))
        #Input layer activation
        self.layers.add_module(f'activation {0}', nn.Softplus())
        for i in range(0, nHiddenLayers):
            #Hidden layer i
            self.layers.add_module(f'fc{i + 1}', nn.Linear(nHiddenNeurons, nHiddenNeurons))
            #Hidden layer i activation
            self.layers.add_module(f'activation {i + 1}', nn.Softplus())
        #output layer
        self.layers.add_module(f'fc{nHiddenLayers + 1}', nn.Linear(nHiddenNeurons, dimOutput))

    #Forward pass through NN
    def forward(self, x):
        tensor = self.layers(x)

        return tensor

    #simulate train/test set
    def generateData(self, X, y, dydx=None):
        #temporary solution
        self.X = X
        self.y = y
        self.dydx = dydx
        self.nSamples = X.shape[0]

        self.alpha = 1/(1 + self.nSamples)
        self.beta = 1 - self.alpha

    #Prepare data
    def prepare(self):
        #Test if tensor
        if torch.is_tensor(self.X) == False:
            self.X = torch.tensor(self.X)
            self.y = torch.tensor(self.y)
            if type(self.dydx) != type(None):
                self.dydx = torch.tensor(self.dydx)

        self.X_scaled, self.y_scaled, self.dydx_scaled, self.lambda_j, \
            self.x_mean, self.x_std, self.y_mean, self.y_std = pre.normalize(self.X, self.y, self.dydx)

        #shaped for NN
        if self.X.dim() == 1:
            nInputs = 1
        else:
            nInputs = self.X.shape[1]

        self.X_scaled = self.X_scaled.view(self.nSamples, nInputs).float()
        self.y_scaled = self.y_scaled.view(self.nSamples, 1).float()
        if type(self.dydx) != type(None):
            self.dydx_scaled = self.dydx_scaled.view(self.nSamples, nInputs).float()

    #Train Net
    def train(self, n_epochs = 3, batch_size=10, lr=0.1):
        alpha = self.alpha
        beta = self.beta

        #check if data is scaled
        if self.X_scaled == None:
            self.prepare()

        if self.differential == False:
            self.loss = Training.trainingLoop(self.X_scaled, self.y_scaled, n_epochs, batch_size, self, lr=lr)
        elif self.differential == True:
            self.loss = Training.diffTrainingLoop(self.X_scaled, self.y_scaled, self.dydx_scaled, n_epochs, batch_size, self, alpha, beta, self.lambda_j, lr=lr)


    #Predict
    def predict(self, X_test, gradients=False):
        #set number of variables
        if X_test.ndim == 1:
            nTest = 1
        else:
            nTest = X_test.shape[1]
        #Test if tensor
        if torch.is_tensor(X_test) == False:
            X_test = torch.tensor(X_test).view(X_test.shape[0], nTest).float()

        #scale
        X_scaled = (X_test - self.x_mean) / self.x_std
        X_scaled = X_scaled.float()
        
        #Predict on scaled X
        y_scaled = self(X_scaled)

        #unscale output from net
        y = self.y_mean + self.y_std * y_scaled

        if gradients == True:
            X_scaled.requires_grad_()
            #predict dydx
            y_derivs_scaled = Training.backprop(X_scaled, self)[0].detach().numpy()
            X_scaled = X_scaled.detach().numpy()
            #unscale dydx
            y_derivs = self.y_std/self.x_std * y_derivs_scaled
            return y.detach().numpy(), y_derivs
        else:
            return y.detach().numpy()

    