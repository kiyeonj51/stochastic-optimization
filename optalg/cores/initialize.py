import numpy as np
from scipy.sparse import diags


def init_zero(data, funcs, tau, reg, info):
    func, grad, hess = funcs
    n, d, C = data['n'], data['d'], data['C']
    beta = np.zeros((d, C))
    vars = {'beta': beta, 'gamma': beta, 'tau': tau}
    if info=="LBFGS":
        vars['s_mem'] = []
        vars['y_mem'] = []
        vars['H'] = diags(np.ones(d*C)) * 1e-3
        vars['r'] = np.sum(grad(beta, data, reg, np.array(range(n))), axis=0)
    return vars


def init_ones(data, params, algo, funcs):
    func, grad, hess = funcs
    n, d, C = data['n'], data['d'], data['C']
    alpha = params['alpha']
    beta = np.zeros((d, C)) * alpha
    vars = {'beta': beta, 'gamma': beta, 'tau': params['tau']}
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