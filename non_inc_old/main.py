
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np
import math
from scipy.io import loadmat, savemat
import argparse
import os
import time
from stl_prim import make_stl_primitives1, make_stl_primitives2
from stl_inf import normal_tree, pruned_tree
from sklearn.model_selection import KFold
import random
import pickle
import sys
sys.path.append("/home/erfan/Documents/University/Projects/Learning_Specifications/python-stl/stl")
from stl import Trace
from stl_inf import best_prim


# ==============================================================================
# ---- Non-Incremental Evaluation ----------------------------------------------
# ==============================================================================
def bdt_evaluation(signals, traces, labels, trees, weights, numtree):
    test = np.zeros(len(signals))
    for i in range(numtree):
        # test = test + weights[i] * np.array([trees[i].classify(trace)
        #                                                 for trace in traces])
        test = test + weights[i] * np.array([trees[i].classify(signal)
                                                        for signal in signals])

    test = np.sign(test)
    return np.count_nonzero(labels - test) / float(len(labels))


# ==============================================================================
# -- Boosted Decision Tree Learning() ------------------------------------------
# ==============================================================================
def boosted_trees(tr_s, tr_t, tr_l, te_s, te_t, te_l, rho_path, primitives, D_t, args):
    depth = args.depth
    numtree = args.numtree
    trees, formulas  = [None] * numtree, [None] * numtree
    prunes = [None] * numtree
    weights, epsilon = [0] * numtree, [0] * numtree
    M = 100

    t = 0
    while t < numtree:
        print('***********************************************************')
        print("Tree {}:".format(t+1))
        if args.prune:
            root_prim, root_impurity, root_rhos = best_prim(tr_s, tr_t, tr_l, rho_path, primitives, D_t, args)
            prune_record = []
            trees[t], prunes[t] = pruned_tree(tr_s, tr_t, tr_l, rho_path, depth, root_prim, root_impurity, root_rhos, primitives, D_t, args, prune_record)
        else:
            trees[t] = normal_tree(tr_s, tr_t, tr_l, rho_path, depth, primitives, D_t, args)
        # rhos = [trees[t].tree_robustness(trace, np.inf) for trace in tr_t]
        formulas[t] = trees[t].get_formula()
        # pred_labels = np.array([trees[t].classify(trace) for trace in tr_t])
        pred_labels = np.array([trees[t].classify(signal) for signal in tr_s])
        for i in range(len(tr_s)):
            if tr_l[i] != pred_labels[i]:
                epsilon[t] = epsilon[t] + D_t[i]

        if epsilon[t] > 0:
            weights[t] = 0.5 * np.log(1/epsilon[t] - 1)
        else:
            weights[t] = M
        print('***********************************************************')
        print("Epsilon:", epsilon[t])
        if epsilon[t] <= 0.5:
            D_t = np.multiply(D_t, np.exp(np.multiply(-weights[t],
                                      np.multiply(tr_l, pred_labels))))
            D_t = np.true_divide(D_t, sum(D_t))
            t = t + 1
        else:
            epsilon[t] = 0


    tr_MCR = 100 * bdt_evaluation(tr_s, tr_t, tr_l, trees, weights, numtree)
    if te_s is None:
        te_MCR = None

    else:
        te_MCR = 100 * bdt_evaluation(te_s, te_t, te_l, trees, weights, numtree)

    return formulas, weights, prunes, tr_MCR, te_MCR


# ==============================================================================
# -- Learn Formula() -----------------------------------------------------------
# ==============================================================================
def learn_formula(tr_s, tr_t, tr_l, primitives, args, te_s = None, te_t = None, te_l = None):
    rho_path    = [np.inf for signal in tr_s]
    D_t         = np.true_divide(np.ones(len(tr_s)), len(tr_s))
    formula, weight, prunes, tr_MCR, te_MCR = boosted_trees(tr_s, tr_t, tr_l, te_s, te_t, te_l,
                                    rho_path, primitives, D_t, args)

    return formula, weight, prunes, tr_MCR, te_MCR


# ==============================================================================
# -- k-fold Learning() ---------------------------------------------------------
# ==============================================================================
def kfold_learning(filename, args):
    mat_data        = loadmat(filename)
    timepoints      = mat_data['t'][0]
    labels          = mat_data['labels'][0]
    signals         = mat_data['data']
    num_dimension   = len(signals[0])
    varnames = ['x_{}'.format(i) for i in range(num_dimension)]
    traces = []
    for i in range(len(signals)):
        data = [signals[i][j] for j in range(num_dimension)]
        trace = Trace(varnames, timepoints, data)
        traces.append(trace)
    signals_shape   = signals.shape
    print('***************************************************************')
    print('(Number of signals, dimension, timepoints):', signals_shape)

    t0 = time.time()
    k_fold = args.fold
    primitives1 = make_stl_primitives1(signals)
    primitives2 = make_stl_primitives2(signals)
    primitives  = primitives1

    if k_fold <= 1:     # No cross-validation
        formula, weight, prunes, tr_MCR, te_MCR = learn_formula(signals, traces, labels, primitives, args)
        print('***************************************************************')
        print("Formula:", formula)
        if args.prune:
            print("Pruning Record:", prunes)
        print("Train MCR:", tr_MCR)
        dt = time.time() - t0
        print('Runtime:', dt)

    else:
        kf = KFold(n_splits = k_fold)
        train_MCR, test_MCR = [], []
        formulas, weights, prunes = [], [], []
        fold_counter = 0
        for train_index, test_index in kf.split(signals):
            tr_signals  = np.array([signals[i] for i in train_index])
            tr_traces   = np.array([traces[i] for i in train_index])
            tr_labels   = np.array([labels[i] for i in train_index])
            te_signals  = np.array([signals[i] for i in test_index])
            te_traces   = np.array([traces[i] for i in test_index])
            te_labels   = np.array([labels[i] for i in test_index])
            fold_counter = fold_counter + 1
            print('***********************************************************')
            print("Fold {}:".format(fold_counter))
            print('***********************************************************')
            formula, weight, prune, tr_MCR, te_MCR = learn_formula(tr_signals, tr_traces, tr_labels,
                                                    primitives, args, te_signals, te_traces, te_labels)
            formulas.append(formula)
            weights.append(weight)
            prunes.append(prune)
            train_MCR.append(tr_MCR)
            test_MCR.append(te_MCR)

        fold_counter = 0
        for f, w, p, tr, te in zip(formulas, weights, prunes, train_MCR, test_MCR):
            fold_counter = fold_counter + 1
            print('***********************************************************')
            print("Fold {}:".format(fold_counter))
            print("Formula:", f)
            print("Weight(s):", w)
            if args.prune:
                print("Pruning Record:", p)
            print("Train MCR:", tr)
            print("Test MCR:", te)
        print('***********************************************************')
        print("Average training error:", sum(train_MCR)/float(k_fold))
        print("Average testing error:", sum(test_MCR)/float(k_fold))
        dt = time.time() - t0
        print('Runtime:', dt)


# ==============================================================================
# -- k-fold Cross-Validation() -------------------------------------------------
# ==============================================================================
def kfold_cross_validation(filename, args):
    mat_data        = loadmat(filename)
    timepoints      = mat_data['t'][0]
    labels          = mat_data['labels'][0]
    signals         = mat_data['data']
    signals_shape   = signals.shape
    print('***************************************************************')
    print('(Number of signals, dimension, timepoints):', signals_shape)

    t0 = time.time()
    k_fold = args.fold
    kf = KFold(n_splits = k_fold)
    candidate_depths = [1, 2, 3]
    candidate_numtrees = [1, 2, 3, 4]
    candidate_k_max = [15, 30]
    candidate_num_particles = [15, 30]

    parameters = []
    for depth in candidate_depths:
        for numtree in candidate_numtrees:
            for k_max in candidate_k_max:
                for num_particles in candidate_num_particles:
                    args.depth = depth
                    args.numtree = numtree
                    args.k_max = k_max
                    args.num_particles = num_particles
                    train_MCR, test_MCR = [], []
                    formulas = []

                    for train_index, test_index in kf.split(signals):
                        tr_signals  = np.array([signals[i] for i in train_index])
                        tr_labels   = np.array([labels[i] for i in train_index])
                        te_signals  = np.array([signals[i] for i in test_index])
                        te_labels   = np.array([labels[i] for i in test_index])
                        formula, weight, tr_MCR, te_MCR = learn_formula(tr_signals, tr_labels,
                                                                te_signals, te_labels, args)
                        formulas.append(formula)
                        train_MCR.append(tr_MCR)
                        test_MCR.append(te_MCR)

                    error = sum(test_MCR)/float(k_fold)
                    parameters.append([error, depth, numtree, k_max, num_particles])

    best_error, best_depth, best_numtree, best_k_max, best_num_particles = min(parameters, key=lambda x: x[0])

    dt = time.time() - t0
    print('***********************************************************')
    print('Best Depth:', best_depth)
    print('Best num_trees:', best_numtree)
    print('Best k_max:', best_k_max)
    print('Best num_particles:', best_num_particles)
    print('Runtime:', dt)
# ==============================================================================
# -- Parse Arguments() ---------------------------------------------------------
# ==============================================================================
def get_argparser():
    parser = argparse.ArgumentParser(formatter_class =
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--action', metavar='A', default='learn', help=
                            """
                            action to take:
                            'learn': builds a classifier for the given training
                            set. The resulting stl formula will be printed.
                            'cv': performs a cross validation test using the
                            given training set.
                            """)
    parser.add_argument('-d', '--depth', metavar='D', type=int,
                        default = 1, help='maximum depth of the decision tree')
    parser.add_argument('-n', '--numtree', metavar='N', type=int,
                        default = 1, help='Number of decision trees')
    parser.add_argument('-k', '--fold', metavar='K', type=int,
                        default=0, help='K-fold cross-validation')
    parser.add_argument('-p', '--prune', metavar='P', type=int,
                        default=0, help='Pruning the tree')
    parser.add_argument('-k_max', '--k_max', metavar='KMAX', type=int,
                        default = 15, help='k_max in pso')
    parser.add_argument('-n_p', '--num_particles', metavar='NP', type=int,
                        default = 15, help='Number of particles in pso')
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
        kfold_learning(get_path(args.file), args)
    else:
        kfold_cross_validation(get_path(args.file), args)
