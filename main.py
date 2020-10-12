
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np
import math
from scipy.io import loadmat, savemat
import argparse
from os import path
import os
from scipy.interpolate import interp1d
from stl_inf import learn_stlinf, perfect_stop, depth_stop
from stl_prim import make_stl_primitives1, make_stl_primitives2, SimpleModel
from stl_linear_optimization import Optimize_Misclass_Gain
# from stlinf import bdt_stlinf_, Traces, _find_best_primitive, perfect_stop, depth_stop
# # from lltinf import perfect_stop, depth_stop, lltinf_, Traces, _find_best_primitive, DTree, SimpleModel
# from stl import robustness
# from llt import make_llt_primitives

# # from stlimpurity import optimize_inf_gain
# # from stloptimization import optimize_inf_gain
# from stl_linear_optimization import Optimize_Misclass_Gain
# # from impurity import optimize_inf_gain
#
#
# from stl import LE, GT




# import glob
# import sys
# try:
#     sys.path.append(glob.glob('../python-stl/stl')[0])
# except IndexError:
#     pass
#
# from stl import Trace

# ==============================================================================
# -- Trace Class -------------------------------------------------------
# ==============================================================================

class Trace(object):
    '''Representation of a system trace.'''

    def __init__(self, variables, timepoints, data, labels, kind='nearest'):
        '''Constructor'''
        self.timepoints = timepoints
        self.labels = labels
        self.traces = data
        self.data = {variable : interp1d(timepoints, var_data, kind=kind)
                            for variable, var_data in zip(variables, data)}
        self.pos_indices, self.neg_indices = self.get_indices()

    def get_indices(self):
        pos_indices, neg_indices = [], []
        for i in range(len(self.labels)):
            if self.labels[i] >= 0:
                pos_indices.append(i)
            else:
                neg_indices.append(i)

        return pos_indices, neg_indices


    def value(self, variable, t):
        '''Returns value of the given signal component at time t.'''
        return self.data[variable](t)

    def values(self, variable, timepoints):
        '''Returns value of the given signal component at desired timepoint.'''
        return self.data[variable](np.asarray(timepoints))

    def __str__(self):
        raise NotImplementedError



# ==============================================================================
# -- Load input traces() -------------------------------------------------------
# ==============================================================================

def load_traces(filename):
    # load data from MAT file
    mat_data    = loadmat(filename)
    timepoints  = list(mat_data['t'][0])
    labels      = list(mat_data['labels'][0])
    num_signals = len(labels)
    varnames    = ['s%d' %i for i in range(num_signals)]
    data        = [mat_data['data'][i] for i in range(num_signals)]
    signals     = Trace(varnames, timepoints, data, labels)
    return signals


# ==============================================================================
# -- Learn Formula() -----------------------------------------------------------
# ==============================================================================

# def bdt_stlinf(traces, pdist, depth=1, inc=0, 
#         optimize_impurity = optimize_inf_gain, stop_condition=None, disp=False):
#     np.seterr(all='ignore')
#     if stop_condition is None:
#         stop_condition = [perfect_stop]

#     return bdt_stlinf_(traces, None, depth, pdist, inc, optimize_impurity,
#                         stop_condition, disp)

# def learn_stlinf(signals, pdist, depth = 1, inc = 0, 
#                     optimize_impurity = optimize_misclass_gain, 
#                     stop_condition = [perfect_stop], disp = False):
#     np.seterr(all='ignore')


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

# def bdt_learn(depth, pdist, inc, disp=False):
#     return lambda data: bdt_stlinf(Traces(*zip(*data)), pdist=pdist, depth=depth,
#                                inc = inc,
#                                stop_condition=[perfect_stop, depth_stop],
#                                disp=disp)


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def learn_formula(matfile, depth, numtree, inc, verbose = True):
    


    ######### DEBUGGING Optimization Problem ################
    signals = load_traces(matfile)
    primitives1 = make_stl_primitives1(signals)
    num_signals = len(signals.labels)
    pdist = np.true_divide(np.ones(num_signals), num_signals)
    prev_rho = None
    disp = False
    optimize_impurity = Optimize_Misclass_Gain(signals, primitives1[0], 1, prev_rho, pdist, disp)
    primitive, impurity = optimize_impurity.get_solution()
    # opt_prims = optimize_inf_gain(signals, primitives1[1].copy(), prim_level, prev_rho, pdist, disp)
    # opt_prims = [optimize_inf_gain(traces, primitive.copy(), True, None, pdist, False) for primitive in primitives1]
    # print(min(opt_prims, key=lambda x: x[1]))



    ######## Initialization ############################
    # intial_rho = None
    # rho_max = 2
    # stop_condition = [perfect_stop, depth_stop]
    # trees, formulas = [None] * numtree 
    # weights, epsilon = [0] * numtree
    # signals = load_traces(matfile)
    # num_signals = len(signals.labels)
    # pdist = np.true_divide(np.ones(num_signals), num_signals)

    # models = [SimpleModel(signal) for signal in traces.signals]

    ######## Boosted Decision Tree ############################

    # for i in range(numtree):
    #     # learn = bdt_learn(depth, pdist, inc, verbose)
    #     # trees.append(learn(zip(*traces.as_list())))
        
    #     trees[i] = learn_stlinf(signals, pdist, depth, inc, initial_rho,
    #                             stop_condition, verbose)
    #     formulas[i] = trees[i].get_formula()
    #     robustnesses = [robustness(formulas[i], model) for model in models]     #### TODO
    #     robust_err = 1 - np.true_divide(np.abs(robustnesses), rho_max)
    #     epsilon[i] = sum(np.multiply(pdist, robust_err))
    #     weights[i] = 0.5 * np.log(1/epsilon[i] - 1)
    #     pred_labels = [trees[i].classify(s) for s in traces.signals] ##### TODO
    #     pdist = np.multiply(pdist, np.exp(np.multiply(-weights[i],
    #                                         np.multiply(labels, pred_labels))))
    #     pdist = np.true_divide(pdist, sum(pdist))
    
    # ###### Evaluation
    # test = np.zeros(numtree)
    # for i in range(numtree):
    #     test = test + np.multiply(weights[i], np.array([trees[i].classify(s) for s in data]))
    
    # test = np.sign(test)
    # # test = np.array([np.sign(sum(np.multiply(weights, trees.classify(s))))
    # #                                                             for s in data])
    
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
