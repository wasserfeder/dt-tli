
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

# def build_single_tree(signals, labels, rho_path, depth, primitives, opt_type, D_t):
#     tree = build_tree(signals, labels, rho_path, depth, primitives, opt_type, D_t)
#     formula = tree.get_formula()
#     print('Formula:', formula)
#     rhos = [tree.tree_robustness(signal, np.inf) for signal in signals]
#     label_MCR = 100 * label_evaluation(signals, labels, tree)
#     robustness_MCR = 100 * robustness_evaluation(labels, rhos)
#     print("Label_based MCR:", label_MCR)
#     print("Robustness_based MCR:", robustness_MCR)





# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

# def build_boosted_trees(signals, labels, rho_path, depth, primitives, opt_type,
#                         D_t, numtree):
#     trees, formulas     = [None] * numtree, [None] * numtree
#     weights, epsilon    = [0] * numtree, [0] * numtree
#
#     for t in range(numtree):
#         trees[t] = build_tree(signals, labels, rho_path, depth, primitives,
#                               opt_type, D_t)
#         formulas[t] = trees[t].get_formula()
#         rhos = [trees[t].tree_robustness(signal, np.inf) for signal in signals]
#         pred_labels = np.array([trees[t].classify(signal) for signal in signals])
#         for i in range(len(signals)):
#             if labels[i] != pred_labels[i]:
#                 epsilon[t] = epsilon[t] + D_t[i]
#         weights[t] = 0.5 * np.log(1/epsilon[t] - 1)
#         D_t = np.multiply(D_t, np.exp(np.multiply(-weights[t], np.multiply(labels, pred_labels))))
#         D_t = np.true_divide(D_t, sum(D_t))
#     print("tree weights:", weights)
#     print("Formulas:", formulas)
#     missclassification_rate = 100 * bdt_evaluation(signals, labels, trees, weights, numtree)
#     print("Misclassification Rate:", missclassification_rate)
#     # fomrula = bdt_get_formula(formulas, weights)
#     # print('Formula:', formula)





# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

# def learn_formula(filename, depth, numtree, inc, opt_type):
#     mat_data        = loadmat(filename)
#     timepoints      = mat_data['t'][0]
#     labels          = mat_data['labels'][0]
#     signals         = mat_data['data']
#     print(signals.shape)
#     print('Number of signals:', len(signals))
#     print('Time points:', len(timepoints))
#
#     t0     = time.time()
#     primitives1 = make_stl_primitives1(signals)
#     primitives2 = make_stl_primitives2(signals)
#     primitives  = primitives1
#     rho_path    = [np.inf for signal in signals]
#     D_t                 = np.true_divide(np.ones(len(signals)), len(signals))
#     dt = time.time() - t0
#     print('Setup time:', dt)
#
#     t0 = time.time()
#     if numtree == 1:
#         build_single_tree(signals, labels, rho_path, depth, primitives,
#                           opt_type, D_t)
#
#     else:
#         build_boosted_trees(signals, labels, rho_path, depth, primitives,
#                             opt_type, D_t, numtree)
#
#     dt = time.time() - t0
#     print('Runtime:', dt)


# ==============================================================================
# ---- Evaluation --------------------------------------------------------------
# ==============================================================================
# Single Tree Evaluation
def label_evaluation(signals, labels, tree):
    labels = np.array(labels)
    predictions = np.array([tree.classify(signal) for signal in signals])
    return np.count_nonzero(labels-predictions)/float(len(labels))

def robustness_evaluation(labels, rhos):
    counter = 0
    for i in range(len(labels)):
        if (labels[i] > 0 and rhos[i] < 0) or (labels[i]<0 and rhos[i]>=0):
            counter = counter + 1
    return counter/float(len(labels))

# Boosted Decision Tree Classification
def bdt_evaluation(signals, labels, trees, weights, numtree):
    test = np.zeros(len(signals))
    for i in range(numtree):
        test = test + weights[i] * np.array([trees[i].classify(signal) for signal in signals])

    test = np.sign(test)
    return np.count_nonzero(labels - test) / float(len(labels))


# ==============================================================================
# -- Single Decision Tree Learning() ------------------------------------------
# ==============================================================================

def build_single_tree(tr_signals, tr_labels, te_signals, te_labels, rho_path, depth, primitives, opt_type, D_t):
    tree = build_tree(tr_signals, tr_labels, rho_path, depth, primitives, opt_type, D_t)
    formula = tree.get_formula()
    tr_MCR = 100 * label_evaluation(tr_signals, tr_labels, tree)
    if te_signals is None:
        te_MCR = None
    else:
        te_MCR = 100 * label_evaluation(te_signals, te_labels, tree)
    return formula, tr_MCR, te_MCR


# ==============================================================================
# -- Boosted Decision Tree Learning() ------------------------------------------
# ==============================================================================

def build_boosted_trees(tr_signals, tr_labels, te_signals, te_labels, rho_path, depth, primitives, opt_type,
                        D_t, numtree):
    trees, formulas     = [None] * numtree, [None] * numtree
    weights, epsilon    = [0] * numtree, [0] * numtree

    for t in range(numtree):
        trees[t] = build_tree(tr_signals, tr_labels, rho_path, depth, primitives,
                              opt_type, D_t)
        formulas[t] = trees[t].get_formula()
        rhos = [trees[t].tree_robustness(signal, np.inf) for signal in tr_signals]
        pred_labels = np.array([trees[t].classify(signal) for signal in tr_signals])
        for i in range(len(tr_signals)):
            if tr_labels[i] != pred_labels[i]:
                epsilon[t] = epsilon[t] + D_t[i]
        weights[t] = 0.5 * np.log(1/epsilon[t] - 1)
        D_t = np.multiply(D_t, np.exp(np.multiply(-weights[t], np.multiply(tr_labels, pred_labels))))
        D_t = np.true_divide(D_t, sum(D_t))
    tr_MCR = 100 * bdt_evaluation(tr_signals, tr_labels, trees, weights, numtree)
    if te_signals is None:
        te_MCR = None
    else:
        te_MCR = 100 * bdt_evaluation(te_signals, te_labels, trees, weights, numtree)

    return formulas, tr_MCR, te_MCR


# ==============================================================================
# -- Learn Formula() -----------------------------------------------------------
# ==============================================================================

def learn_formula(tr_signals, tr_labels, te_signals, te_labels, depth, numtree, inc, opt_type):
    t0          = time.time()
    primitives1 = make_stl_primitives1(tr_signals)
    primitives2 = make_stl_primitives2(tr_signals)
    primitives  = primitives1
    rho_path    = [np.inf for signal in tr_signals]
    D_t         = np.true_divide(np.ones(len(tr_signals)), len(tr_signals))
    dt          = time.time() - t0
    print('Setup time:', dt)

    t0 = time.time()
    if numtree == 1:
        formula, tr_MCR, te_MCR = build_single_tree(tr_signals, tr_labels, te_signals, te_labels, rho_path, depth, primitives,
                          opt_type, D_t)

    else:
        formula, tr_MCR, te_MCR = build_boosted_trees(tr_signals, tr_labels, te_signals, te_labels, rho_path, depth, primitives,
                            opt_type, D_t, numtree)

    dt = time.time() - t0
    print('Runtime:', dt)
    return formula, tr_MCR, te_MCR


# ==============================================================================
# -- k-fold Cross Validation() -------------------------------------------------
# ==============================================================================
def cross_validation(filename, depth, numtree, inc, k_fold, opt_type):
    mat_data        = loadmat(filename)
    timepoints      = mat_data['t'][0]
    labels          = mat_data['labels'][0]
    signals         = mat_data['data']
    print(signals.shape)
    print('Number of signals:', len(signals))
    print('Time points:', len(timepoints))
    if k_fold <= 1:
        formula, tr_MCR, te_MCR = learn_formula(signals, labels, None, None, depth, numtree, inc, opt_type)
        print("Formula:", formula)
        print("MCR:", tr_MCR)
    else:
        kf = KFold(n_splits = k_fold)
        train_MCR, test_MCR = [], []
        formulas = []
        for train_index, test_index in kf.split(signals):
            tr_signals = [signals[i] for i in train_index]
            tr_labels = [labels[i] for i in train_index]
            te_signals = [signals[i] for i in test_index]
            te_labels = [labels[i] for i in test_index]
            formula, tr_MCR, te_MCR = learn_formula(tr_signals, tr_labels, te_signals,
                                      te_labels, depth, numtree, inc, opt_type)
            formulas.append(formula)
            train_MCR.append(tr_MCR)
            test_MCR.append(te_MCR)

        for f, tr, te in zip(formulas, train_MCR, test_MCR):
            print("Formula:", f)
            print("Train MCR:", tr)
            print("Test MCR:", te)
        print("Average training error:", sum(train_MCR)/float(k_fold))
        print("Average testing error:", sum(test_MCR)/float(k_fold))

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
    cross_validation(get_path(args.file), args.depth, args.numtree, args.inc,
                  args.fold ,args.optimization)
    # learn_formula(get_path(args.file), args.depth, args.numtree, args.inc,
    #               args.optimization)
