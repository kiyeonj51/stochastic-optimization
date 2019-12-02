from optalg.cores.loss import *
from optalg.cores.utils import *
from optalg.cores.data import *
from optalg.cores.optimize import GD, AGD, SGD, MBSGD, SVRG, LBFGS
import sys
# sys.path.append('..')
# from cores.data import *
# from cores.optimize import *
# from cores.loss import *
# from cores.utils import *
# from settings.setup_algorithm import *
# from settings.setting_optimization import *
from optalg.cores.initialize import *

# Example setting

params = {
    "data_name": DataMNIST,
    "seed": 200,
    "reg": 1e-4,
    "tau": 0,
    "stepsize": .1,
    "max_iter": 10,
    "a": 0.1,
    "b": 0.9,
    "init_method": init_zero,
    "alpha": 1.,
    "is_mcr": True,
    "is_numgrad": True,
    "is_save": False,
    "m": 10,
    "batch": 1000,
}

funcs = (func_lr_mc,  grad_lr_mc, None)

opt_algorithms = [
    ("GD", "+", GD(linesearch=constantstep, max_iter=params['max_iter'], reg=params['reg'], stepsize=params['stepsize'], info='GD')),
    ("AGD", "*", AGD(linesearch=constantstep, max_iter=params['max_iter'], reg=params['reg'],stepsize=params['stepsize'], info='AGD')),
    ("SGD", ".", SGD(linesearch=decreasingstep2, max_iter=params['max_iter'], reg=params['reg']*.1,stepsize=params['stepsize'], info='SGD')),
    ("Minibatch-SGD", "d", MBSGD(linesearch=decreasingstep2, max_iter=params['max_iter'], reg=params['reg'],stepsize=params['stepsize'], batch=params['batch'], info='MBSGD')),
    ("SVRG", "+", SVRG(linesearch=constantstep, max_iter=params['max_iter'], reg=params['reg'],stepsize=params['stepsize']*.1, m=params['m'], info='SVRG')),
    ("L-BFGS","*", LBFGS(linesearch=btls, max_iter=params['max_iter'], reg=params['reg'],stepsize=params['stepsize'], info='LBFGS')),
]


plots = [
    {"ylabel": "tr_errors", "xlabel": "times",
     "yscale":"log", "title": "train loss", "xlim": False},
    {"ylabel": "tr_errors", "xlabel": "iteration",
     "yscale":"log", "title": "train loss", "xlim": False},
    {"ylabel": "tr_errors", "xlabel": "num_grads",
     "yscale":"log", "title": "train loss", "xlim": False},

    {"ylabel": "te_errors", "xlabel": "iteration",
     "yscale":"log", "title": "test loss", "xlim": False},

    {"ylabel": "mcrs", "xlabel": "times",
     "yscale":"log", "title": "mis classification rate", "xlim": False},
    {"ylabel": "mcrs", "xlabel": "iteration",
     "yscale":"log", "title": "mis classification rate", "xlim": False},
    {"ylabel": "mcrs", "xlabel": "num_grads",
     "yscale": "log", "title": "mis classification rate", "xlim": False},
]


dataset = params['data_name']()
data = dataset.load_data()

results = []
for name, marker, algorithm in opt_algorithms:
    print(f'{name}')
    res = algorithm.solve(data=data, funcs=funcs)
    res["marker"] = marker
    res["name"]=name
    if params["is_mcr"]:
        res["mcrs"] = missclassrate(data['test'], res['betas'])
    if params["is_numgrad"]:
        res["num_grad"] = algorithm.num_grad(data['train'])
    results.append(res)


# collect results
outputs = {'tr_errors': [res['tr_errors'] for res in results],
           'te_errors': [res['te_errors'] for res in results],
           'times': [res['times'] for res in results],
           'markers': [res['marker'] for res in results],
           'names': [res['name'] for res in results],
           'is_save': params['is_save']}
if params["is_mcr"]:
    outputs['mcrs'] = [res['mcrs'] for res in results]
if params["is_numgrad"]:
    outputs['num_grads'] = [res['num_grad'] for res in results]

# plot results
for plot in plots:
    plotresult(plot, outputs)
