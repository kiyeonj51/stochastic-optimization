import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags


def btls(beta, func, grad, delta, data, reg, samples, linesearch):
    t = 1.
    a = linesearch['alpha']
    b = linesearch['beta']
    while np.sum(func(beta + t * delta, data, reg, samples)) \
            > np.sum(func(beta, data, reg, samples)) \
            + a * t * ((np.sum(grad(beta, data, reg, samples), axis=0).flatten('F')).dot(delta.flatten('F'))):
        t = b * t
    return t


def constantstep(beta, func, grad, desc, data, reg, samples, linesearch):
    return linesearch['stepsize']


def decreasingstep1(beta, func, grad, desc, data, reg, samples, linesearch):
    return linesearch['stepsize'] / (linesearch['iter'] + 1)


def decreasingstep2(beta, func, grad, desc, data, reg, samples, linesearch):
    return linesearch['stepsize'] / np.sqrt(linesearch['iter'] + 1)


def plotresult(plot,outputs, file_name=''):
    ylabel = plot['ylabel']
    xlabel = plot['xlabel']
    yscale = plot['yscale']
    title = plot['title']
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    if xlabel == "iteration":
        for i in range(len(outputs[ylabel])):
            plt.plot(outputs[ylabel][i], label=outputs['names'][i])
    else:
        for i in range(len(outputs[ylabel])):
            plt.plot(outputs[xlabel][i], outputs[ylabel][i], label=outputs['names'][i])
    plt.title(title)
    plt.yscale(yscale)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if plot['xlim']:
        xmin = np.inf
        for xs in outputs[xlabel]:
            xmin = min(xmin, xs[-1])
        plt.xlim((0, xmin))
    if outputs['is_save']:
        plt.savefig(file_name+'_'+plot['xlabel']+'_'+plot['ylabel']+'.png')

    plt.show()


def missclassrate(test, betas):
    X, Y = test['X'], test['Y']
    mcr = []
    for iter in range(len(betas)):
        cnt = np.sum(np.argmin(X.dot(betas[iter]), axis=1) != Y)
        mcr.append(cnt / Y.shape[0])
    return mcr