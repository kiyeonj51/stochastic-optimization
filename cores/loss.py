import numpy.linalg as la
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# func_logistic_regression_multiclass
def func_lr_mc(beta, data, params, samples):
    reg = params['reg']
    X = data['X'][samples, :]
    Y = data['Y'][samples]
    n = Y.shape[0]
    loss1 = 0
    for idx in range(n):
        loss1 += np.sum(X[idx, :].dot(beta[:, Y[idx]]))
    loss2 = np.sum(np.log(np.sum(np.exp(-X.dot(beta)), axis=1)))
    loss = (1. / n) * (loss1 + loss2)
    regularizer = reg/2. * la.norm(beta, 'fro') ** 2
    return loss, regularizer


# grad_logistic_regression_multiclass
def grad_lr_mc(beta, data, params, samples):
    X = data['X'][samples, :]
    Y = data['Y'][samples]
    n = Y.shape[0]
    onehot = OneHotEncoder(categories='auto')
    E = onehot.fit_transform(data['Y'].reshape(-1, 1)).T[:, samples]
    # E = data['E'][:, samples]
    W = (np.exp(-X.dot(beta)).T / np.sum(np.exp(-X.dot(beta)), axis=1)).T
    grad_loss = (1. / n) * ((E.dot(X)).T - X.T.dot(W))
    grad_reg = params['reg'] * beta
    return grad_loss, grad_reg