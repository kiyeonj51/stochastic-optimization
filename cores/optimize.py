from time import *
import numpy as np
import numpy.linalg as la
from tqdm import tqdm
from tqdm import trange
from abc import ABC, abstractmethod
from cores.loss import *
from cores.utils import *
from scipy.sparse import diags


class Optimize(ABC):
    def __init__(self, data, funcs, params, algo):
        self.data = data
        self.funcs = funcs
        self.params = params
        self.algo = algo

    def descent(self):
        linesearch = self.algo['linesearch']
        funcs = self.funcs
        train = self.data['train']
        test = self.data['test']
        algo = self.algo
        params = self.params
        vars = params['init_method'](train, params, algo, funcs)
        func, grad, hess = funcs
        beta = vars['beta']
        res = {}
        betas, losses, regs, te_errors, elapsed_times = [], [], [], [], []

        betas.append(beta)
        loss, reg = func(beta, train, params, np.array(range(train['n'])))
        te_error = func(beta, test, params, np.array(range(test['n'])))[0]
        losses.append(loss)
        regs.append(reg)
        te_errors.append(te_error)
        elapsed_times.append( 0 )
        with trange(algo['max_iter']) as t:
            for iter in t:
                start = time()
                vars = self.update(vars, funcs, train, linesearch, iter, algo, params)
                end = time()
                beta = vars['beta']
                betas.append(beta)
                loss , reg = func(beta, train, params, np.array(range(train['n'])))
                te_error = func(beta, test, params, np.array(range(test['n'])))[0]
                losses.append(loss)
                regs.append(reg)
                te_errors.append(te_error)
                elapsed_times.append(end - start)
                t.set_description('tr_error : {:3.6f}, reg : {:.7e}, te_error : {:3.6f} '.format(loss, reg, te_error))
                if iter % int(algo['max_iter']/5) == 0:
                    t.write(' ')
        res = {
            'betas':betas, 'beta': beta, 'tr_errors': losses,
            'regs': regs, 'te_errors': te_errors, 'times': list(np.cumsum(elapsed_times))
        }
        return res

    @abstractmethod
    def update(self, vars, funcs, data, linesearch, iter, algo, params):
        raise NotImplementedError



class GD(Optimize):
    def update(self, vars, funcs, data, linesearch, iter, algo, params):
        func, grad, hess = funcs
        beta = vars['beta']
        n = data['n']
        samples = np.array(range(n))
        desc = -np.sum(grad(beta, data, params, samples), axis=0)
        eta = linesearch(beta, func, grad, desc, data, params, samples, algo, iter)
        beta = beta + eta * desc
        vars['beta'] = beta
        return vars
    def num_grad(self):
        n = self.data['train']['n']
        cnts = [0] + list(range(self.algo['max_iter']))
        return cnts


class SGD(Optimize):
    def update(self, vars, funcs, data, linesearch, iter, algo, params):
        func, grad, hess = funcs
        beta = vars['beta']
        n = data['n']
        samples = np.random.randint(n, size=1)
        desc = -np.sum(grad(beta, data, params, samples), axis=0)
        eta = linesearch(beta, func, grad, desc, data, params, samples, algo, iter)
        beta = beta + eta * desc
        vars['beta'] = beta
        return vars
    def num_grad(self):
        n = self.data['train']['n']
        cnts = list(range(self.algo['max_iter']))
        cnts = [0] + [cnt / n for cnt in cnts]
        return cnts


class MBSGD(Optimize):
    def update(self, vars, funcs, data, linesearch, iter, algo, params):
        func, grad, hess = funcs
        beta = vars['beta']
        n = data['n']
        samples = np.random.randint(n, size=algo['batch'])
        desc = -np.sum(grad(beta, data, params, samples), axis=0)
        eta = linesearch(beta, func, grad, desc, data, params, samples, algo, iter)
        beta = beta + eta * desc
        vars['beta'] = beta
        return vars
    def num_grad(self):
        n = self.data['train']['n']
        cnts = list(range(self.algo['max_iter']))
        cnts = [0] + [cnt * self.algo['batch'] / n for cnt in cnts]
        return cnts


class SVRG(Optimize):
    def update(self, vars, funcs, data, linesearch, iter, algo, params):
        func, grad, hess = funcs
        beta = vars['beta']
        n = data['n']
        m = algo['m']
        full_grad = np.sum(grad(beta, data, params, np.array(range(n))), axis=0)
        beta_0 = beta
        # x = np.random.randint(m, size=1)
        for iter in range(m):
            samples = np.random.randint(n, size=1)
            desc = -(np.sum(grad(beta, data, params, samples), axis=0)
                     - np.sum(grad(beta_0, data, params, samples), axis=0)
                     + full_grad)
            eta = linesearch(beta, func, grad, desc, data, params, samples, algo, iter)
            beta = beta + eta * desc
        vars['beta'] = beta
        return vars

    def num_grad(self):
        n = self.data['train']['n']
        cnts = list(range(self.algo['max_iter']))
        cnts = [0] + [cnt * (n + self.algo['m']) / n for cnt in cnts]
        return cnts


class LBFGS(Optimize):
    # def init_vars(self):
    #     numpts = self.data['train']['X'].shape[0]
    #     func, grad, hess =self.funcs
    #     initbeta = np.zeros(self.data['train']['Z'].shape)
    #     vars = {
    #         'beta': initbeta,
    #         's_mem': [],
    #         'y_mem': [],
    #         'H': diags(np.ones(initbeta.shape[0] * initbeta.shape[1])) * 1e-3,
    #         'r': np.sum(grad(initbeta, self.data['train'], self.params), axis=0)
    #             }
    #     vars['p'] = -vars['r']
    #     return vars

    def update(self, vars, funcs, data, linesearch, iter, algo, params):
        func, grad, hess = funcs
        n = data['n']
        m = algo['m']
        d = data['d']
        C = data['C']
        samples = np.array(range(n))
        beta, H, s_mem, y_mem = vars['beta'], vars['H'], vars['s_mem'], vars['y_mem']
        gradient = np.sum(grad(beta, data, params, samples), axis=0)
        p = -self.direction(gradient, s_mem, y_mem, iter, H, m)
        alpha = linesearch(beta, func, grad, p, data, params, samples, algo, iter)
        beta_old = beta
        beta = beta + alpha * p
        if iter > m:
            s_mem, y_mem = s_mem[:, 1:], y_mem[:, 1:]
        s = beta - beta_old
        gradient_old = gradient
        gradient = np.sum(grad(beta, data, params, samples), axis=0)
        y = gradient - gradient_old
        if iter == 0:
            s_mem, y_mem = s.flatten('F'), y.flatten('F')
        else:
            s_mem, y_mem = np.column_stack((s_mem, s.flatten('F'))), np.column_stack((y_mem, y.flatten('F')))
        gamma = (s.flatten('F').dot(y.flatten('F')) / y.flatten('F').dot(y.flatten('F')))
        H = gamma * diags(np.ones(d * C))
        var = {'beta': beta, 'H': H, 's_mem': s_mem, 'y_mem': y_mem}
        return var

    def direction(self, gradient, s_mem, y_mem, iter, H, m):
        q = gradient.flatten('F')
        if iter < m:
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



