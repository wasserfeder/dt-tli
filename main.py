
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
import matplotlib.pyplot as plt


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def build_single_tree(signals, labels, rho_path, depth, primitives, opt_type, D_t):
    tree = build_tree(signals, labels, rho_path, depth, primitives, opt_type, D_t)
    formula = tree.get_formula()
    print('Formula:', formula)
    rhos = [tree.tree_robustness(signal, np.inf) for signal in signals]
    label_MSR = 100 * label_evaluation(signals, labels, tree)
    robustness_MSR = 100 * robustness_evaluation(labels, rhos)
    print("Label_based MSR:", label_MSR)
    print("Robustness_based MSR:", robustness_MSR)


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def build_boosted_trees(signals, labels, rho_path, depth, primitives, opt_type,
                        D_t, numtree):
    # rho_max             = 10000
    trees, formulas     = [None] * numtree, [None] * numtree
    weights, epsilon    = [0] * numtree, [0] * numtree

    for t in range(numtree):
        trees[t] = build_tree(signals, labels, rho_path, depth, primitives,
                              opt_type, D_t)
        formulas[t] = trees[t].get_formula()
        rhos = [trees[t].tree_robustness(signal, np.inf) for signal in signals]
        pred_labels = np.array([trees[t].classify(signal) for signal in signals])
        for i in range(len(signals)):
            if labels[i] != pred_labels[i]:
                epsilon[t] = epsilon[t] + D_t[i]
        weights[t] = 0.5 * np.log(1/epsilon[t] - 1)
        D_t = np.multiply(D_t, np.exp(np.multiply(-weights[t], np.multiply(labels, pred_labels))))
        D_t = np.true_divide(D_t, sum(D_t))
    print("tree weights:", weights)
    missclassification_rate = 100 * bdt_evaluation(signals, labels, trees, weights, numtree)
    print("Misclassification Rate:", missclassification_rate)
    # fomrula = bdt_get_formula(formulas, weights)
    # print('Formula:', formula)



# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def learn_formula(filename, depth, numtree, inc, opt_type):
    mat_data        = loadmat(filename)
    timepoints      = mat_data['t'][0]
    labels          = mat_data['labels'][0]
    signals         = mat_data['data']
    print(signals.shape)
    print('Number of signals:', len(signals))
    print('Time points:', len(timepoints))

    t0     = time.time()
    primitives1 = make_stl_primitives1(signals)
    primitives2 = make_stl_primitives2(signals)
    primitives  = primitives1
    rho_path    = [np.inf for signal in signals]
    D_t                 = np.true_divide(np.ones(len(signals)), len(signals))
    dt = time.time() - t0
    print('Setup time:', dt)

    t0 = time.time()
    if numtree == 1:
        build_single_tree(signals, labels, rho_path, depth, primitives,
                          opt_type, D_t)

    else:
        build_boosted_trees(signals, labels, rho_path, depth, primitives,
                            opt_type, D_t, numtree)

    dt = time.time() - t0
    print('Runtime:', dt)


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
# -- Parse Arguments() ---------------------------------------------------------
# ==============================================================================

def get_argparser():
    parser = argparse.ArgumentParser(formatter_class =
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('optimization', choices=['milp', 'pso'], nargs='?',
                        default='pso', help='optimization type')
    parser.add_argument('-d', '--depth', metavar='D', type=int,
                        default=1, help='maximum depth of the decision tree')
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
    learn_formula(get_path(args.file), args.depth, args.numtree, args.inc,
                  args.optimization)
