
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
from stl_inf import build_tree
from stl_inc_inf import build_inc_tree
from stl_syntax import GT
from sklearn.model_selection import KFold
import random


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================
def inc_tree(tr_s, tr_l, te_s, te_l, rho_path, depth, primitives, opt_type, D_t):
    timepoints = len(tr_s[0][0])
    trees, weights = [], []
    formulas = []
    for t in range(2, timepoints):
        tr_s_partial = tr_s[:,:,:t]
        tree = build_inc_tree(tr_s_partial, tr_l, rho_path, depth, primitives, opt_type, D_t)
        if tree is not None:
            trees.append(tree)
            formulas.append(tree.get_formula())
            rhos = [tree.tree_robustness(signal, np.inf) for signal in tr_s_partial]
            pred_labels = np.array([tree.classify(signal) for signal in tr_s_partial])
            epsilon = 0
            for i in range(len(tr_s_partial)):
                if tr_l[i] != pred_labels[i]:
                    epsilon = epsilon + D_t[i]

            weights.append(0.5 * np.log(1/epsilon - 1))
            D_t = np.multiply(D_t, np.exp(np.multiply(-weights[-1],
                                          np.multiply(tr_l, pred_labels))))
            D_t = np.true_divide(D_t, sum(D_t))

    numtree = len(trees)
    tr_MCR = 100 * bdt_evaluation(tr_s, tr_l, trees, weights, numtree)
    if te_s is None:
        te_MCR = None

    else:
        te_MCR = 100 * bdt_evaluation(te_s, te_l, trees, weights, numtree)

    return formulas, tr_MCR, te_MCR



# ==============================================================================
# ---- Evaluation --------------------------------------------------------------
# ==============================================================================
# Single Tree Evaluation
def label_evaluation(signals, labels, tree):
    labels      = np.array(labels)
    predictions = np.array([tree.classify(signal) for signal in signals])
    return np.count_nonzero(labels-predictions)/float(len(labels))

def robustness_evaluation(labels, rhos):
    counter = 0
    for i in range(len(labels)):
        if (labels[i] > 0 and rhos[i] < 0) or (labels[i] < 0 and rhos[i] >= 0):
            counter = counter + 1
    return counter/float(len(labels))

# Boosted Decision Tree Classification
def bdt_evaluation(signals, labels, trees, weights, numtree):
    test = np.zeros(len(signals))
    for i in range(numtree):
        test = test + weights[i] * np.array([trees[i].classify(signal)
                                                        for signal in signals])

    test = np.sign(test)
    return np.count_nonzero(labels - test) / float(len(labels))


# ==============================================================================
# -- Single Decision Tree Learning() ------------------------------------------
# ==============================================================================
def single_tree(tr_s, tr_l, te_s, te_l, rho_path, primitives, D_t, args):
    depth = args.depth
    tree    = build_tree(tr_s, tr_l, rho_path, depth, primitives, D_t, args)
    formula = tree.get_formula()
    tr_MCR  = 100 * label_evaluation(tr_s, tr_l, tree)
    if te_s is None:
        te_MCR = None

    else:
        te_MCR = 100 * label_evaluation(te_s, te_l, tree)

    return formula, tr_MCR, te_MCR


# ==============================================================================
# -- Boosted Decision Tree Learning() ------------------------------------------
# ==============================================================================
def boosted_trees(tr_s, tr_l, te_s, te_l, rho_path, primitives, D_t, args):
    depth = args.depth
    numtree = args.numtree
    trees, formulas  = [None] * numtree, [None] * numtree
    weights, epsilon = [0] * numtree, [0] * numtree

    for t in range(numtree):
        trees[t] = build_tree(tr_s, tr_l, rho_path, depth, primitives, D_t, args)
        rhos = [trees[t].tree_robustness(signal, np.inf) for signal in tr_s]
        formulas[t] = trees[t].get_formula()
        pred_labels = np.array([trees[t].classify(signal) for signal in tr_s])
        for i in range(len(tr_s)):
            if tr_l[i] != pred_labels[i]:
                epsilon[t] = epsilon[t] + D_t[i]

        if epsilon[t] > 0:
            weights[t] = 0.5 * np.log(1/epsilon[t] - 1)
        else:
            weights[t] = 0
        D_t = np.multiply(D_t, np.exp(np.multiply(-weights[t],
                                      np.multiply(tr_l, pred_labels))))
        D_t = np.true_divide(D_t, sum(D_t))

    tr_MCR = 100 * bdt_evaluation(tr_s, tr_l, trees, weights, numtree)
    if te_s is None:
        te_MCR = None

    else:
        te_MCR = 100 * bdt_evaluation(te_s, te_l, trees, weights, numtree)

    return formulas, tr_MCR, te_MCR


# ==============================================================================
# -- Learn Formula() -----------------------------------------------------------
# ==============================================================================
def learn_formula(tr_s, tr_l, te_s, te_l, args):
    inc         = args.inc
    numtree     = args.numtree
    primitives1 = make_stl_primitives1(tr_s)
    primitives2 = make_stl_primitives2(tr_s)
    primitives  = primitives1 + primitives2
    rho_path    = [np.inf for signal in tr_s]
    D_t         = np.true_divide(np.ones(len(tr_s)), len(tr_s))

    if not inc:     # Non-incremental version
        if numtree == 1:   # No boosted decision trees
            formula, tr_MCR, te_MCR = single_tree(tr_s, tr_l, te_s, te_l,
                                    rho_path, primitives, D_t, args)

        else:
            formula, tr_MCR, te_MCR = boosted_trees(tr_s, tr_l, te_s, te_l,
                                    rho_path, primitives, D_t, args)
    else:         # Incremental version
        formula, tr_MCR, te_MCR = inc_tree(tr_s, tr_l, te_s, te_l, rho_path,
                                            depth, primitives, opt_type, D_t)


    return formula, tr_MCR, te_MCR


# ==============================================================================
# -- k-fold Learning() ---------------------------------------------------------
# ==============================================================================
def kfold_learning(filename, args):
    mat_data        = loadmat(filename)
    timepoints      = mat_data['t'][0]
    labels          = mat_data['labels'][0]
    signals         = mat_data['data']
    signals_shape   = signals.shape
    # Shuffling the data:
    # temp = list(zip(signals, labels))
    # random.shuffle(temp)
    # res1, res2 = zip(*temp)
    # signals = list(res1)
    # labels = list(res2)
    print('***************************************************************')
    print('(Number of signals, dimension, timepoints):', signals_shape)

    t0 = time.time()
    k_fold = args.fold

    if k_fold <= 1:     # No cross-validation
        formula, tr_MCR, te_MCR = learn_formula(signals, labels, None,None,args)
        print('***************************************************************')
        print("Formula:", formula)
        print("Train MCR:", tr_MCR)
        dt = time.time() - t0
        print('Runtime:', dt)

    else:
        kf = KFold(n_splits = k_fold)
        train_MCR, test_MCR = [], []
        formulas = []
        for train_index, test_index in kf.split(signals):
            tr_signals  = np.array([signals[i] for i in train_index])
            tr_labels   = np.array([labels[i] for i in train_index])
            te_signals  = np.array([signals[i] for i in test_index])
            te_labels   = np.array([labels[i] for i in test_index])
            formula, tr_MCR, te_MCR = learn_formula(tr_signals, tr_labels,
                                                    te_signals, te_labels, args)
            formulas.append(formula)
            train_MCR.append(tr_MCR)
            test_MCR.append(te_MCR)

        fold_counter = 0
        for f, tr, te in zip(formulas, train_MCR, test_MCR):
            fold_counter = fold_counter + 1
            print('***********************************************************')
            print("Fold {}:".format(fold_counter))
            print("Formula:", f)
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
    candidate_depths = [1, 2, 3, 4]
    candidate_numtrees = [1, 2, 3, 4, 5]
    candidate_k_max = [15, 30, 50]
    candidate_num_particles = [15, 20, 50]

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
                        formula, tr_MCR, te_MCR = learn_formula(tr_signals, tr_labels,
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
    parser.add_argument('-i', '--inc', metavar='I', type=int,
                        default=0, help='Incremental or Non-incremental')
    parser.add_argument('-k', '--fold', metavar='K', type=int,
                        default=0, help='K-fold cross-validation')
    parser.add_argument('-k_max', '--k_max', metavar='KMAX', type=int,
                        default = 15, help='k_max in pso')
    parser.add_argument('-n_p', '--num_particles', metavar='NP', type=int,
                        default = 15, help='Number of particles in pso')
    parser.add_argument('optimization', choices=['milp', 'pso'], nargs='?',
                        default='pso', help='optimization type')
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
