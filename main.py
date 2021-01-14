
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
from stl_syntax import GT
from sklearn.model_selection import KFold


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================
def inc_tree(tr_s, tr_l, te_s, te_l, rho_path, depth, primitives, opt_type, D_t):
    tree = build_inc_tree(tr_s, tr_l, rho_path, depth, primitives, opt_type, D_t)
    formula = tree.get_formula()
    # tr_MCR  = 100 * label_evaluation(tr_s, tr_l, tree)
    # if te_s is None:
    #     te_MCR = None
    #
    # else:
    #     te_MCR = 100 * label_evaluation(te_s, te_l, tree)
    #
    # return formula, tr_MCR, te_MCR








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
def single_tree(tr_s, tr_l, te_s, te_l, rho_path, depth, primitives, opt_type,
                                                                        D_t):
    tree    = build_tree(tr_s, tr_l, rho_path, depth, primitives, opt_type, D_t)
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
def boosted_trees(tr_s, tr_l, te_s, te_l, rho_path, depth, primitives, opt_type,
                                                                D_t, numtree):
    trees, formulas  = [None] * numtree, [None] * numtree
    weights, epsilon = [0] * numtree, [0] * numtree

    for t in range(numtree):
        trees[t] = build_tree(tr_s, tr_l, rho_path, depth, primitives, opt_type,
                   D_t)
        rhos = [trees[t].tree_robustness(signal, np.inf) for signal in tr_s]
        formulas[t] = trees[t].get_formula()
        pred_labels = np.array([trees[t].classify(signal) for signal in tr_s])
        for i in range(len(tr_s)):
            if tr_l[i] != pred_labels[i]:
                epsilon[t] = epsilon[t] + D_t[i]

        weights[t] = 0.5 * np.log(1/epsilon[t] - 1)
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
    depth       = args.depth
    numtree     = args.numtree
    inc         = args.inc
    opt_type    = args.optimization
    primitives1 = make_stl_primitives1(tr_s)
    primitives2 = make_stl_primitives2(tr_s)
    primitives  = primitives1
    rho_path    = [np.inf for signal in tr_s]
    D_t         = np.true_divide(np.ones(len(tr_s)), len(tr_s))

    if not inc:     # Non-incremental version
        if numtree == 1:   # No boosted decision trees
            formula, tr_MCR, te_MCR = single_tree(tr_s, tr_l, te_s, te_l,
                                    rho_path, depth, primitives, opt_type, D_t)

        else:
            formula, tr_MCR, te_MCR = boosted_trees(tr_s, tr_l, te_s, te_l,
                                    rho_path, depth, primitives, opt_type, D_t,
                                    numtree)
    else:         # Incremental version
        formula, tr_MCR, te_MCR = inc_tree(tr_s, tr_l, te_s, te_l, rho_path,
                                            depth, primitives, opt_type, D_t)


    return formula, tr_MCR, te_MCR


# ==============================================================================
# -- k-fold Cross Validation() -------------------------------------------------
# ==============================================================================
def cross_validation(filename, args):
    mat_data        = loadmat(filename)
    timepoints      = mat_data['t'][0]
    labels          = mat_data['labels'][0]
    signals         = mat_data['data']
    print('***************************************************************')
    print('(Number of signals, dimension, timepoints):', signals.shape)
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
# -- Parse Arguments() ---------------------------------------------------------
# ==============================================================================
def get_argparser():
    parser = argparse.ArgumentParser(formatter_class =
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--depth', metavar='D', type=int,
                        default=1, help='maximum depth of the decision tree')
    parser.add_argument('-n', '--numtree', metavar='N', type=int,
                        default=0, help='Number of decision trees')
    parser.add_argument('-i', '--inc', metavar='I', type=int,
                        default=0, help='Incremental or Non-incremental')
    parser.add_argument('-k', '--fold', metavar='K', type=int,
                        default=0, help='K-fold cross-validation')
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
    cross_validation(get_path(args.file), args)
