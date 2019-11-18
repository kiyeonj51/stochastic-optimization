# from cores.data import *
# from cores.loss import *
# # dataset = DataNews()
# dataset = DataDigits()
# data = dataset.load_data()
# n = data['train']['Y'].shape[0]
# B = np.random.rand(data['train']['X'].shape[1],len(set(data['train']['Y'])))
# param={'reg':.1}
# samples = np.array(range(n))
# # batch = 1
# # samples = np.random.randint(n, size=batch)
# l,r = func_lr_mc(B, data['train'],samples,reg=.1)
# gl,gr = grad_lr_mc(B, data['train'],param,samples)
# # print("")



import sys
sys.path.append('..')
from cores.data import *
from cores.optimize import *
from cores.loss import *
from cores.utils import *
# from settings.setup_algorithm import *
from settings.setting_gd_sgd_mbsgd_svrg import *
import argparse


def main():
    # seed number
    np.random.seed(params["seed"])

    # load/generate dataset
    dataset = eval(data_name)()
    data = dataset.load_data()

    # optimize
    results = []
    mcrs = []
    num_grads = []
    for algo in algorithms:
        print('\n'+algo['name'])
        opt = eval(algo['method'].upper())(data=data, funcs=funcs, params=params, algo=algo)
        res = opt.descent()
        if params["is_mcr"]:
            mcr = missclassrate(data['test'], res['betas'])
            res["mcrs"] = mcr
        if params["is_numgrad"]:
            num_grad = opt.num_grad()
            res["num_grad"]=num_grad
        results.append(res)


    # plot results
    tr_errors = [res['tr_errors'] for res in results]
    te_errors = [res['te_errors'] for res in results]
    if params["is_mcr"]:
        mcrs = [res['mcrs'] for res in results]
    if params["is_numgrad"]:
        num_grads = [res['num_grad'] for res in results]
    times = [res['times'] for res in results]
    descs = [algo['desc'] for algo in algorithms]
    markers = [algo['marker'] for algo in algorithms]

    summary = {
        "time": times,
        "train_error": tr_errors,
        "test_error": te_errors,
        "desc": descs,
        "marker": markers,
        "is_save": params["is_save"],
        "data_name": data_name
    }
    if params["is_mcr"]:
        summary["mcr"] = mcrs
    if params["is_numgrad"]:
        summary["num_grad"] = num_grads

    for plot in plots:
        plotresult(plot, summary, file_name)
        # plotresult(x=summary[plot['x']], y=summary[plot['y']],  names=names, plot=plot)


if __name__=="__main__":
    main()


# archive codes
# parser = argparse.ArgumentParser()
# parser.add_argument("-dataset", dest="dataset")
# parser.add_argument("-seed", nargs=1, dest="seed",type=int)
# parser.add_argument("-reg", dest="reg", type=float)
# parser.add_argument("-tau", dest="tau", type=float)
# parser.add_argument("-stepsize", dest="stepsize", type=float)
# parser.add_argument("-max_iter", dest="max_iter", type=int)
# parser.add_argument("-m", dest="m", nargs='+', type=int)
# parser.add_argument("-a", dest="a",type=float)
# parser.add_argument("-b", dest="b", type=float)
# parser.add_argument("-mcr", dest="mcr", action='store_true')
# args = parser.parse_args()
# # setup params
# reg = args.reg
# tau = args.tau
# stepsize = args.stepsize
# max_iter = args.max_iter
# a = args.a
# b = args.b
# markevery = int(max_iter / 5)