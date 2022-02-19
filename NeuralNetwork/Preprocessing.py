import torch

#Preprocessing
def normalize(X, y, dydx = None):
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

    return X, y, dydx, lambda_j, X_mean.item(), X_std.item(), y_mean.item(), y_std.item()

def unscale(X, y, dydx, X_mean, X_std, y_mean, y_std):
    X = X_mean + X_std * X
    y = y_mean + y_std * y

    dydx = y_std / X_std * dydx

    return X, y, dydx