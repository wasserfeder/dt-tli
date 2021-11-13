import numpy as np
import math
from nn_train import learn_weights


def get_horizons(trees, formulas):
    horizons = np.zeros(len(formulas))
    for i in range(len(formulas)):
        tree = trees[i]
        children = tree.get_children()
        tree_horizon = []
        for child in children:
            tree_horizon.append(child.high)
        horizons[i] = max(tree_horizon)
    return horizons



def get_weights(signals, labels, trees, formulas):
    signal_horizon = len(signals[0][0])
    len_signals = len(signals)
    formula_horizons = get_horizons(trees, formulas)
    for j in range(len(formulas)):
        print("Formula: ", formulas[j])
        print("Horizon: ", formula_horizons[j])

    # Structure of the nn_output: {'t': [[trees], [weights]]}, which includes all time steps
    nn_output = {}
    prev_active_formulas = []
    prev_active_trees = np.array([])

    rhos = None
    for t in range(signal_horizon):
        max_ind = np.searchsorted(formula_horizons, t)
        if (max_ind == 0) and t < formula_horizons[max_ind]:
            active_formulas = []
            active_trees = np.array([])
        elif max_ind >= len(formula_horizons):
            active_formulas = formulas
            active_trees = trees
        elif t == formula_horizons[max_ind]:
            active_formulas = formulas[:max_ind+1]
            active_trees = np.array(trees[:max_ind + 1])
        else:
            active_formulas = formulas[:max_ind]
            active_trees = np.array(trees[:max_ind])

        signals_par = signals[:,:,:t+1]
        if len(active_formulas) == 0:
            nn_output[t] = []
        elif active_formulas == prev_active_formulas:
            nn_output[t] = nn_output[t-1]
        else:
            rhos = compute_rhos(signals_par, rhos, active_trees, prev_active_trees)
            weights = learn_weights(rhos, labels)
            nn_output[t] = [active_trees, weights]

        prev_active_formulas = active_formulas
        prev_active_trees = active_trees

    return nn_output


def compute_rhos(signals, rhos, trees, prev_trees):
    new_trees = np.setdiff1d(trees, prev_trees)
    for i in range(len(new_trees)):
        rho_path = [np.inf for signal in signals]
        new_rhos = np.array(new_trees[i].tree_robustness(signals, rho_path))
        new_rhos = np.reshape(new_rhos, (len(new_rhos), 1))
        if rhos is None:
            rhos = new_rhos
        else:
            rhos = np.append(rhos, new_rhos, axis=1)
    return rhos
