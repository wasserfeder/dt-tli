''' 
parsed down version of dt-tli non-inc branch for the purposes of generating formulas using trajectories in the HAL-Suite 

TODO:
'''
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np
import sys

sys.dont_write_bytecode = True
from .stl_inf import normal_tree, pruned_tree,best_prim
from pytelo.stl import Trace

from hal_suite import set_package_seed

set_package_seed(1)

# ==============================================================================
# ---- Non-Incremental Evaluation ----------------------------------------------
# ==============================================================================
def bdt_evaluation(signals, traces, labels, trees, weights, numtree):
    M = 100
    test = np.zeros(len(signals))
    m_counter = 0
    for i in range(numtree):
        if weights[i] == M:
            m_counter +=1
            index = i
    if m_counter == 1:
        test = test + weights[index] * np.array([trees[index].classify(signal,trace)
                                                        for (signal,trace) in zip(signals,traces)])
    else:
        for i in range(numtree):
            test = test + weights[i] * np.array([trees[i].classify(signal,trace)
                                                        for (signal,trace) in zip(signals,traces)])

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
        formulas[t] = trees[t].get_formula()
        pred_labels = np.array([trees[t].classify(signal,trace) for (signal,trace) in zip(tr_s,tr_t)])
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


