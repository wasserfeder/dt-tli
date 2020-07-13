
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np
import math
from scipy.io import loadmat, savemat
import argparse
from os import path
import os
from stlinf import bdt_stlinf_, Traces, _find_best_primitive, perfect_stop, depth_stop
# from lltinf import perfect_stop, depth_stop, lltinf_, Traces, _find_best_primitive, DTree, SimpleModel
from stl import robustness
from llt import make_llt_primitives
from stlprim import make_stl_primitives1, make_stl_primitives2, SimpleModel
# from stlimpurity import optimize_inf_gain
from stloptimization import optimize_inf_gain
# from impurity import optimize_inf_gain


from stl import LE, GT
# ==============================================================================
# -- Load input traces() -------------------------------------------------------
# ==============================================================================

def load_traces(filename):
    # load data from MAT file
    mat_data =  loadmat(filename)
    # add time dimension to signals' data
    dims = list(mat_data['data'].shape)
    dims[1] += 1
    data = np.zeros(dims)
    data[:, :dims[1]-1, :] = mat_data['data']
    data[:, dims[1]-1, :] = mat_data['t']
    # create list of labels (classes: positive, negative)
    if 'labels' in mat_data:
        labels = list(mat_data['labels'][0])
    else:
        labels = None
    # create data structure of traces
    return Traces(signals=data, labels=labels)


# ==============================================================================
# -- Learn Formula() -----------------------------------------------------------
# ==============================================================================

def bdt_stlinf(traces, pdist, depth=1,
 inc=0, optimize_impurity = optimize_inf_gain,
               stop_condition=None, disp=False):
    np.seterr(all='ignore')
    if stop_condition is None:
        stop_condition = [perfect_stop]

    return bdt_stlinf_(traces, None, depth, pdist, inc, optimize_impurity,
                        stop_condition, disp)


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def bdt_learn(depth, pdist, inc, disp=False):
    return lambda data: bdt_stlinf(Traces(*zip(*data)), pdist=pdist, depth=depth,
                               inc = inc,
                               stop_condition=[perfect_stop, depth_stop],
                               disp=disp)


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def learn_formula(matfile, depth, numtree, inc, verbose = True):
    traces = load_traces(matfile)
    data = zip(*traces.as_list())
    labels = np.array(traces.labels)
    pdist = np.true_divide(np.ones(traces.m), traces.m)
    trees, weights, formulas  = [], [], []
    epsilon = []
    models = [SimpleModel(signal) for signal in traces.signals]
    r_max = 2


    ######### DEBUGGING ################
    primitives1 = make_stl_primitives1(traces.signals)
    opt_prims = [optimize_inf_gain(traces, primitive.copy(), True, None, pdist, False) for primitive in primitives1]
    print(min(opt_prims, key=lambda x: x[1]))
    # print(primitives2[0].args[0].args[0].args[0].op)
    # print(primitives2[0].args[0]._op)
    # optimize_inf_gain(traces, primitives2[0], None, pdist)

    # if primitives[0].args[0].args[0].args[0].op == LE:
    #     print("Hi")
    # print(primitives[0].args[0].args[0].index)

    ########################


    # for i in range(numtree):
    #     learn = bdt_learn(depth, pdist, inc, verbose)
    #     trees.append(learn(zip(*traces.as_list())))
    #     formulas.append(trees[i].get_formula())
    #     robustnesses = [robustness(formulas[i], model) for model in models]
    #     robust_err = 1 - np.true_divide(np.abs(robustnesses), r_max)
    #     epsilon = sum(np.multiply(pdist, robust_err))
    #     weights.append(0.5 * np.log(1/epsilon - 1))
    #     pred_labels = [trees[i].classify(s) for s in traces.signals]
    #     pdist = np.multiply(pdist, np.exp(np.multiply(-weights[i],
    #                                         np.multiply(labels, pred_labels))))
    #     pdist = np.true_divide(pdist, sum(pdist))
    #
    # ###### Evaluation
    # test = np.zeros(numtree)
    # for i in range(numtree):
    #     test = test + np.multiply(weights[i], np.array([trees[i].classify(s) for s in data]))
    #
    # test = np.sign(test)
    # # test = np.array([np.sign(sum(np.multiply(weights, trees.classify(s))))
    # #                                                             for s in data])
    #
    # print(np.count_nonzero(labels - test) / float(len(labels)))


# ==============================================================================
# -- Parse Arguments() ---------------------------------------------------------
# ==============================================================================

def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('action', choices=['learn'], nargs='?',
                        default='learn', help=
                            """
                            action to take:
                            'learn': builds a classifier for the given training
                            set. The resulting stl formula will be printed.
                            """)
    parser.add_argument('-d', '--depth', metavar='D', type=int,
                        default=1, help='maximum depth of the decision tree')
    # parser.add_argument('--out-perm', metavar='f', default=None,
    #                     help='if specified, saves the cross validation permutation into f')
    parser.add_argument('-n', '--numtree', metavar='N', type=int,
                        default=0, help='Number of decision trees')
    parser.add_argument('-i', '--inc', metavar='I', type=int,
                        default=0, help='Incremental or Non-incremental')
    parser.add_argument('file', help='.mat file containing the data')
    return parser




def get_path(f):
    return path.join(os.getcwd(), f)
# ==============================================================================
# -- global variables and functions---------------------------------------------
# ==============================================================================

if __name__ == '__main__':
    args = get_argparser().parse_args()
    if args.action == 'learn':
        learn_formula(get_path(args.file), args.depth, args.numtree, args.inc)
    else:
        raise Exception("Action not implemented")
