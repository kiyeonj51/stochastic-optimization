import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

def init_zero(data,params, algo, funcs):
    func, grad, hess = funcs
    n, d, C = data['n'], data['d'], data['C']
    beta = np.zeros((d, C))
    vars = {'beta': beta}
    if algo['method']=="lbfgs":
        vars['s_mem'] = []
        vars['y_mem'] = []
        vars['H'] = diags(np.ones(d*C)) * 1e-3
        vars['r'] = np.sum(grad(beta, data, params, np.array(range(n))), axis=0)
    return vars


def init_ones(data, params, algo, funcs):
    func, grad, hess = funcs
    n, d, C = data['n'], data['d'], data['C']
    alpha = params['alpha']
    beta = np.zeros((d, C)) * alpha
    vars = { 'beta': beta}
    if algo['method'] == 'lbfgs':
        vars['s_mem'] = []
        vars['y_mem'] = []
        vars['H'] = diags(np.ones(d*C)) * 1e-3
        vars['r'] = np.sum(grad(beta, data, params, np.array(range(n))), axis=0)
    return vars


def init_rand_normal(data, params, algo, funcs):
    func, grad, hess = funcs
    n, d, C = data['n'], data['d'], data['C']
    alpha = params['alpha']
    beta = np.random.rand(d,C)*alpha
    vars = {'beta': beta}
    if algo['method'] == 'lbfgs':
        vars['s_mem'] = []
        vars['y_mem'] = []
        vars['H'] = diags(np.ones(d*C)) * 1e-3
        vars['r'] = np.sum(grad(beta, data, params, np.array(range(n))), axis=0)
    return vars


def btls(beta, func, grad, desc, data, params, samples, algo, iter):
    a, b = params['a'], params['b']
    t = 1.
    while np.sum(func(beta + t * desc, data, params, samples)) \
            > np.sum(func(beta, data, params, samples)) \
            + a * t * ((np.sum(grad(beta, data, params, samples), axis=0).flatten('F')).dot(desc.flatten('F'))):
        t = b * t
        # print(t)
    return t


def constantstep(beta, func, grad, desc, data, params, samples, algo, iter):
    return algo['stepsize']


def decreasingstep1(beta, func, grad, desc, data, params, samples, algo, iter):
    return algo['stepsize'] / (iter + 1)


def decreasingstep2(beta, func, grad, desc, data, params, samples, algo, iter):
    return algo['stepsize'] / np.sqrt(iter + 1)


def plotresult(plot,summary, file_name):
    ylabel = plot['ylabel']
    xlabel = plot['xlabel']
    yscale = plot['yscale']
    title = plot['title']
    fig = plt.figure()
    if xlabel == "iteration":
        for i in range(len(summary[ylabel])):
            plt.plot(summary[ylabel][i], label=summary['desc'][i])
    else:
        for i in range(len(summary[ylabel])):
            plt.plot(summary[xlabel][i], summary[ylabel][i], label=summary['desc'][i])
    plt.title(title)
    plt.yscale(yscale)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plot['xlim']:
        xmin = np.inf
        for xs in summary[xlabel]:
            xmin = min(xmin, xs[-1])
        plt.xlim((0, xmin))
    if summary['is_save']:
        plt.savefig(file_name+'_'+plot['xlabel']+'_'+plot['ylabel']+'.png')
    plt.show()


def missclassrate(test, betas):
    X, Y = test['X'], test['Y']
    mcr = []
    for iter in range(len(betas)):
        cnt = np.sum(np.argmin(X.dot(betas[iter]), axis=1) != Y)
        mcr.append(cnt / Y.shape[0])
    return mcr