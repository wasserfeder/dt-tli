
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np
import math
from scipy.io import loadmat, savemat
import argparse
import os
import time
from stl_prim import make_stl_primitives1
from stl_inf import build_tree



# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def learn_formula(filename, depth, numtree, inc):
    ######### DEBUGGING Optimization Problem ################
    mat_data        = loadmat(filename)
    timepoints      = mat_data['t'][0]
    labels          = mat_data['labels'][0]
    signals         = mat_data['data']
    print(signals.shape)
    print('Number of signals:', len(signals))
    print('Time points:', len(timepoints))

    t_begin     = time.time()
    primitives1 = make_stl_primitives1(signals)
    rho_path    = [np.inf for signal in signals]
    # [primitive, obj] = find_best_primitive(signals, labels, primitives1, rho_path)

    tree = build_tree(signals, labels, timepoints, depth, primitives1, rho_path)
    formula = tree.get_formula()
    print('Formula:', formula)

    run_time = time.time() - t_begin
    print('Runtime:', run_time)




    # trees, formulas = [None] * numtree, [None] * numtree        # for boosted
    # weights, epsilon = [0] * numtree, [0] * numtree             # for boosted
    # D_t = np.true_divide(np.ones(len(signals)), len(signals))   # for boosted
    # intial_rho = None
    # rho_max = 2
    # stop_condition = [perfect_stop, depth_stop]
    # models = [SimpleModel(signal) for signal in traces.signals]

    # for t in range(numtree):
    #     trees[t] = learn_stlinf(signals, pdist, depth, inc, initial_rho,
    #                             stop_condition, verbose)
    #     formulas[t] = trees[t].get_formula()
    #     robustnesses = [robustness(formulas[t], model) for model in models]     #### TODO
    #     robust_err = 1 - np.true_divide(np.abs(robustnesses), rho_max)
    #     epsilon[t] = sum(np.multiply(pdist, robust_err))
    #     weights[t] = 0.5 * np.log(1/epsilon[t] - 1)
    #     pred_labels = [trees[t].classify(s) for s in traces.signals] ##### TODO
    #     pdist = np.multiply(pdist, np.exp(np.multiply(-weights[t],
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
    return os.path.join(os.getcwd(), f)
# ==============================================================================
# -- global variables and functions---------------------------------------------
# ==============================================================================

if __name__ == '__main__':
    args = get_argparser().parse_args()
    if args.action == 'learn':
        learn_formula(get_path(args.file), args.depth, args.numtree, args.inc)
    else:
        raise Exception("Action not implemented")
