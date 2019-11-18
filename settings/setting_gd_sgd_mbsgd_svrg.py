from cores.loss import *
from cores.utils import *

data_name = "DataNews"

params = {
    "seed": 100,
    "reg": 5.,
    "tau": 0,
    "stepsize": .01,
    "max_iter": 5,
    "a": 0.1,
    "b": 0.9,
    "init_method": init_ones,
    "alpha": 1.,
    "is_mcr": True,
    "is_numgrad": False,
    "is_save": True,

}

# params = {
#     "seed": 100,
#     "reg": 1e-7,
#     "tau": 0,
#     "stepsize": 1.,
#     "max_iter": 50,
#     "a": 0.1,
#     "b": 0.9,
#     "init_method": init_ones,
#     "alpha": 10.,
#     "is_mcr": True,
#     "is_numgrad": False,
#     "is_save": True,
#
# }

file_name = []
for key, val in params.items():
    if key == "init_method":
        val = val.__name__
    if key in ["seed", "reg", "stepsize", "max_iter","init_param"]:
        file_name.append(key)
        file_name.append(str(val))
file_name = data_name+'_'+'_'.join(file_name)


algorithms = [
    {"method": "lbfgs", "m":5, "linesearch":btls, "desc": "lbfgs_m_5",
     "marker":"+", "stepsize":params['stepsize'],
     "max_iter":params["max_iter"]*10},

    {"method": "lbfgs", "m":10, "linesearch":btls, "desc": "lbfgs_m_10",
     "marker":"*", "stepsize":params['stepsize'],
     "max_iter": params["max_iter"]*10},

{"method": "lbfgs", "m":20, "linesearch":btls, "desc": "lbfgs_m_20",
     "marker":"*", "stepsize":params['stepsize'],
     "max_iter": params["max_iter"]*10},

    {"method": "gd", "linesearch": btls, "desc": "gd",
     "marker": "+", "stepsize":params['stepsize'],
     "max_iter": params["max_iter"]},
    #
    # {"method": "sgd", "linesearch": decreasingstep1, "desc": "sgd",
    #  "marker": "*", "stepsize":params['stepsize'],
    #  "max_iter":params["max_iter"]},
    # {"method": "svrg", "linesearch": constantstep, "desc":"svrg",
    #  "marker": ".", "stepsize" : params['stepsize'],
    #  "max_iter":params["max_iter"],"m": 10},
    # {"method": "mbsgd", "linesearch":decreasingstep1,"desc":"mbsgd_10",
    #  "marker": "d", "stepsize":params['stepsize'],
    #  "max_iter":int(params["max_iter"]/10), "batch":10},
    # {"method": "mbsgd", "linesearch":decreasingstep1,"desc":"mbsgd_20",
    #  "marker": "d", "stepsize":params['stepsize'],
    #  "max_iter":int(params["max_iter"]/20), "batch":20}
]

for algo in algorithms:
    name = []
    for key, val in algo.items():
        if (key != "marker") and (key != "desc"):
            if key == "linesearch":
                val = val.__name__
            name.append(key)
            name.append(str(val))
    algo['name'] = '_'.join(name)

funcs = (func_lr_mc,  grad_lr_mc, None)



plots = [
    {"ylabel": "train_error", "xlabel": "time",
     "yscale":"log", "title": "train loss", "xlim": True},
    {"ylabel": "train_error", "xlabel": "iteration",
     "yscale":"log", "title": "train loss", "xlim": False},
    # {"ylabel": "train_error", "xlabel": "num_grad",
    #  "yscale":"log", "title": "train loss", "xlim": True},

    {"ylabel": "test_error", "xlabel": "time",
     "yscale":"log", "title": "test loss", "xlim": True},
    {"ylabel": "test_error", "xlabel": "iteration",
     "yscale":"log", "title": "test loss", "xlim": False},
    # {"ylabel": "test_error", "xlabel": "num_grad",
    #  "yscale":"log", "title": "test loss", "xlim": True},

    {"ylabel": "mcr", "xlabel": "time",
     "yscale":"log", "title": "mis classification rate", "xlim": True},
    {"ylabel": "mcr", "xlabel": "iteration",
     "yscale":"log", "title": "mis classification rate", "xlim": False},
    # {"ylabel": "mcr", "xlabel": "num_grad",
    #  "yscale":"log", "title": "mis classification rate", "xlim": True},
]
