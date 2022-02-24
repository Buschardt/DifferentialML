import torch
import torch.nn.functional as F
import torch.optim as optim

#Regular training loop
def trainingLoop(X, y, n_epochs, batch_size, NeuralNet, lr=0.001):
    optimizer = optim.Adam(NeuralNet.parameters(), lr=lr)

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


##Differential Learning Functions
#Get gradients of NN w.r.t. inputs
def backprop(X, net):
    gradients = torch.autograd.grad(net(X), X, grad_outputs=torch.ones(X.shape[0], 1), create_graph=True, retain_graph=True, allow_unused=True)
    return gradients

#Define custom loss function with differential regularization
def diffReg(labels, predictions, derivLabels, derivPredictions, alpha, beta, lambda_j):
    loss_values = alpha * F.mse_loss(labels, predictions)
    loss_derivs = beta * F.mse_loss(derivLabels * lambda_j, derivPredictions * lambda_j)
    return loss_values + loss_derivs

#Training loop with differential regularization
def diffTrainingLoop(X, y_values, y_derivs, n_epochs, batch_size, NeuralNet, alpha, beta, lambda_j=1, lr=0.001):
    optimizer = optim.Adam(NeuralNet.parameters(), lr=lr)

    for epoch in range(n_epochs):

        permutation = torch.randperm(X.size()[0])

        for i in range(0,X.size()[0], batch_size):
            indicies = permutation[i:i+batch_size]
            X_batch, y_values_batch, y_derivs_batch = X[indicies], y_values[indicies], y_derivs[indicies]
            NeuralNet.zero_grad()
            output = NeuralNet(X_batch)
            X_batch.requires_grad_()
            predDerivs = backprop(X_batch, NeuralNet) #compute derivatives of net
            loss = diffReg(y_values_batch, output, y_derivs_batch, predDerivs[0], alpha, beta, lambda_j)
            loss.backward()
            optimizer.step()
        print(loss)