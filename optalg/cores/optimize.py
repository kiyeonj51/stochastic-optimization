from time import *
import numpy as np
import numpy.linalg as la
from tqdm import tqdm
from tqdm import trange
from abc import ABC, abstractmethod
# from cores.loss import *
# from cores.utils import *
from scipy.sparse import diags
from sklearn.preprocessing import OneHotEncoder
from optalg.cores.utils import *
from optalg.cores.initialize import *


class OPT(ABC):
    def __init__(self, linesearch=btls,max_iter=100, stepsize=1e-2, reg=0., tau=0, seed=200,
                 btls_alpha=.1, btls_beta=.9, m=10, batch=1000, info='GD'):
        self.linesearch=linesearch
        self.btls_alpha = btls_alpha
        self.btls_beta = btls_beta
        self.max_iter=max_iter
        self.stepsize=stepsize
        self.reg = reg
        self.tau = tau
        self.seed = seed
        self.m = m
        self.batch = batch
        self.info = info

    def solve(self, data, funcs, init_method=init_zero):
        # params
        linesearch = {"method": self.linesearch, "alpha": self.btls_alpha, "beta": self.btls_beta, "stepsize": self.stepsize}
        info = self.info
        tau = self.tau
        reg = self.reg
        m = self.m
        batch = self.batch
        max_iter = self.max_iter
        train = data['train']
        test = data['test']

        # initialization
        onehot = OneHotEncoder(categories='auto')
        train['E'] = onehot.fit_transform(train['Y'].reshape(-1, 1)).T
        variable = init_method(train, funcs, tau, reg, info)
        betas, tr_errors, regularizers, te_errors, elapsed_times = [], [], [], [], []

        func, grad, hess = funcs
        beta = variable['beta']
        betas.append(beta)
        tr_error, regularizer = func(beta, train, reg, np.array(range(train['n'])))
        te_error = func(beta, test, reg, np.array(range(test['n'])))[0]
        tr_errors.append(tr_error)
        regularizers.append(regularizer)
        te_errors.append(te_error)
        elapsed_times.append(0)

        # optimization
        with trange(max_iter-1) as t:
            for iteration in t:
                linesearch["iter"] = iteration
                start = time()
                variable = self.update(variable, train, funcs, reg, linesearch, m=m, batch=batch)
                end = time()
                beta = variable['beta']
                tr_error, regularizer = func(beta, train, reg, np.array(range(train['n'])))
                te_error = func(beta, test, reg, np.array(range(test['n'])))[0]
                betas.append(beta)
                tr_errors.append(tr_error)
                regularizers.append(regularizer)
                te_errors.append(te_error)
                elapsed_times.append(end - start)
                t.set_description('tr_error : {:3.6f}, regularizer : {:.7e}, te_error : {:3.6f} '.format(tr_error, regularizer, te_error))
                if (iteration > 0) and (iteration % int(max_iter/5) == 0):
                    t.write(' ')
        res = {
            'betas': betas, 'beta': beta,
            'tr_errors': tr_errors, 'regularizers': regularizers, 'te_errors': te_errors,
            'times': list(np.cumsum(elapsed_times))
        }
        return res

    @abstractmethod
    def update(self, variable, data, funcs, reg, linesearch, **kwargs):
        raise NotImplementedError


class GD(OPT):
    def update(self, variable, data, funcs, reg, linesearch, **kwargs):
        func, grad, hess = funcs
        beta = variable['beta']
        n = data['n']
        samples = np.array(range(n))
        delta = -np.sum(grad(beta, data, reg, samples), axis=0)
        eta = linesearch['method'](beta, func, grad, delta, data, reg, samples, linesearch)
        beta = beta + eta * delta
        variable['beta'] = beta
        return variable

    def num_grad(self, train):
        n = train['n']
        counts = list(np.array(range(self.max_iter)) * n)
        return counts


class AGD(OPT):
    def update(self, variable, data, funcs, reg, linesearch, **kwargs):
        func, grad, hess = funcs
        beta = variable['beta']
        gamma = variable['gamma']
        tau = variable['tau']
        tau_next = (1 + np.sqrt(1 + 4 * np.square(tau))) / 2
        ratio = (1 - tau) / tau_next

        n = data['n']
        samples = np.array(range(n))

        delta = -np.sum(grad(beta, data, reg, samples), axis=0)
        eta = linesearch['method'](beta, func, grad, delta, data, reg, samples, linesearch)
        gamma_next = beta + eta * delta
        beta_next = (1 - ratio) * gamma_next + ratio * gamma

        variable['beta'] = beta_next
        variable['gamma'] = gamma_next
        variable['tau'] = tau_next
        return variable

    def num_grad(self, train):
        n = train['n']
        counts = list(np.array(range(self.max_iter)) * (2*n))
        return counts


class SGD(OPT):
    def update(self, variable, data, funcs, reg, linesearch, **kwargs):
        func, grad, hess = funcs
        beta = variable['beta']
        n = data['n']
        samples = np.random.choice(n, size=1, replace=False)
        delta = -np.sum(grad(beta, data, reg, samples), axis=0)
        eta = linesearch['method'](beta, func, grad, delta, data, reg, samples, linesearch)
        beta = beta + eta * delta
        variable['beta'] = beta
        return variable

    def num_grad(self, train):
        counts = list(range(self.max_iter))
        return counts


class MBSGD(OPT):
    def update(self, variable, data, funcs, reg, linesearch, **kwargs):
        func, grad, hess = funcs
        beta = variable['beta']
        n = data['n']
        samples = np.random.choice(n, size=kwargs['batch'], replace=False)
        delta = -np.sum(grad(beta, data, reg, samples), axis=0)
        eta = linesearch['method'](beta, func, grad, delta, data, reg, samples, linesearch)
        beta = beta + eta * delta
        variable['beta'] = beta
        return variable

    def num_grad(self, train):
        counts = list(range(self.max_iter))
        counts = [cnt * self.batch for cnt in counts]
        return counts


class SVRG(OPT):
    def update(self, variable, data, funcs, reg, linesearch, **kwargs):
        func, grad, hess = funcs
        beta = variable['beta']
        n = data['n']
        m = kwargs['m']
        full_grad = np.sum(grad(beta, data, reg, np.array(range(n))), axis=0)
        beta_0 = beta
        for it in range(m):
            samples = np.random.choice(n, size=1, replace=False)
            delta = -(np.sum(grad(beta, data, reg, samples), axis=0)
                     - np.sum(grad(beta_0, data, reg, samples), axis=0)
                     + full_grad)
            eta = linesearch['method'](beta, func, grad, delta, data, reg, samples, linesearch)
            beta = beta + eta * delta
        variable['beta'] = beta
        return variable

    def num_grad(self, train):
        n = train['n']
        counts = list(range(self.max_iter))
        counts = [cnt * (n + self.m) for cnt in counts]
        return counts


class LBFGS(OPT):
    def update(self, variable, data, funcs, reg, linesearch, **kwargs):
        func, grad, hess = funcs
        n = data['n']
        m = kwargs['m']
        d = data['d']
        C = data['C']
        iteration = linesearch['iter']
        samples = np.array(range(n))
        beta, H, s_mem, y_mem = variable['beta'], variable['H'], variable['s_mem'], variable['y_mem']
        gradient = np.sum(grad(beta, data, reg, samples), axis=0)
        p = -self.direction(gradient, s_mem, y_mem, iteration, H, m)
        alpha = linesearch['method'](beta, func, grad, p, data, reg, samples, linesearch)
        beta_old = beta
        beta = beta + alpha * p
        if iteration > m:
            s_mem, y_mem = s_mem[:, 1:], y_mem[:, 1:]
        s = beta - beta_old
        gradient_old = gradient
        gradient = np.sum(grad(beta, data, reg, samples), axis=0)
        y = gradient - gradient_old
        if iteration == 0:
            s_mem, y_mem = s.flatten('F'), y.flatten('F')
        else:
            s_mem, y_mem = np.column_stack((s_mem, s.flatten('F'))), np.column_stack((y_mem, y.flatten('F')))
        gamma = (s.flatten('F').dot(y.flatten('F')) / y.flatten('F').dot(y.flatten('F')))
        H = gamma * diags(np.ones(d * C))
        var = {'beta': beta, 'H': H, 's_mem': s_mem, 'y_mem': y_mem}
        return var

    def direction(self, gradient, s_mem, y_mem, iteration, H, m):
        q = gradient.flatten('F')
        if iteration < m:
            r = H.dot(q)
        else:
            alpha = np.zeros(m)
            for i in range(m - 1, -1, -1):
                alpha[i] = ((1. / s_mem[:, i].dot(y_mem[:, i])) * (s_mem[:, i].dot(q)))
                q = q - alpha[i] * y_mem[:, i]
            r = H.dot(q)
            for i in range(m):
                beta = (1. / s_mem[:, i].dot(y_mem[:, i])) * (y_mem[:, i].dot(r))
                r = r + s_mem[:, i] * (alpha[i] - beta)
        r = r.reshape(gradient.shape, order='F')
        return r

    def num_grad(self, train, **kwargs):
        n = train['n']
        m = self.m
        k = self.max_iter
        counts = list(np.array(range(m))*n) + list(np.array(range(m, k))*(2*m) + (m-1)*n)
        return counts



