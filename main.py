# TODO:
'''
* Two types of growing trees: 1) only based on partial signals, 2) based on
partial signals and prediction of the future of signals

* For the grow criteria: 1) sum of the distances between signals, 2) max, min,
and other variations of the distances

'''

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np, math, argparse, os, time, random, pickle, sys
from scipy.io import loadmat, savemat
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from stl_prim import make_stl_primitives1
from stl_inf import inc_tree
from nn_preparation import get_weights
sys.path.append("/home/erfan/Documents/University/Projects/Learning_Specifications/python-stl/stl")

# ==============================================================================
# ---- Incremental Evaluation ----------------------------------------------
# ==============================================================================
def inc_evaluation(signals, labels, nn_output):
    signal_horizon = len(signals[0][0])
    MCR_vote, MCR_rho = np.zeros(signal_horizon), np.zeros(signal_horizon)

    # return MCR_vote, MCR_rho
    return None, None


# ==============================================================================
# -- Distance between two signals() --------------------------------------------
# ==============================================================================
def signal_distance(pos_signal, neg_signal):
    distance = 0
    for j in range(len(pos_signal)):
        distance += (pos_signal[j] - neg_signal[j])**2

    return np.sqrt(distance)


# ==============================================================================
# -- check growth condition() ------------------------------------------
# ==============================================================================
def check_growth(pos_signals, neg_signals):
    timepoints = len(pos_signals[0][0])
    mean_distances = np.zeros(timepoints)
    len_pos, len_neg = len(pos_signals), len(neg_signals)
    for t in range(timepoints):
        pos_signals_par = pos_signals[:, :, t]
        neg_signals_par = neg_signals[:, :, t]
        distance_array = []
        for p in range(len_pos):
            pos_signal = pos_signals_par[p]
            for n in range(len_neg):
                neg_signal = neg_signals_par[n]
                distance = signal_distance(pos_signal, neg_signal)
                distance_array.append(distance)
        mean_distances[t] = np.mean(distance_array)

    zero_ind = []
    for i in range(1, timepoints-1):
        if ((mean_distances[i] > mean_distances[i-1]) and (mean_distances[i] > mean_distances[i+1])) or ((mean_distances[i] < mean_distances[i-1]) and (mean_distances[i] < mean_distances[i+1])):
            zero_ind.append(i)

    decision_times = [zero_ind[0]]
    for i in range(len(zero_ind)-1):
        decision_times.append(int(0.5*(zero_ind[i] + zero_ind[i+1])))
        decision_times.append(zero_ind[i+1])
    decision_times.append(timepoints-1)
    print("Decision Times:", decision_times)
    return decision_times


# ==============================================================================
# -- Boosted Decision Tree Learning() ------------------------------------------
# ==============================================================================
def boosted_trees(tr_s, tr_l, te_s, te_l, rho_path, primitives, args):
    tr_pos_ind, tr_neg_ind = np.where(tr_l > 0)[0], np.where(tr_l <= 0)[0]
    pos_signals, neg_signals = signals[tr_pos_ind], signals[tr_neg_ind]
    # decision_times = check_growth(pos_signals, neg_signals)
    ##### For naval:
    decision_times = [12, 15, 18, 26, 34, 37, 41, 60]

    depth, tree_limit = args.depth, 5
    trees, formulas = [], []
    for t in decision_times:
        tree = None
        tree_counter = 0
        tr_s_par = tr_s[:,:,:t]
        while (tree is None and tree_counter < tree_limit):
            tree = inc_tree(tr_s_par, tr_l, rho_path, depth, primitives, args)
            tree_counter += 1
            if tree is not None:
                formula = tree.get_formula()
                pred_labels = np.array([tree.classify(s) for s in tr_s_par])
                epsilon = tr_l != pred_labels
                if sum(epsilon)/len(tr_s) <= 0.5:
                    trees.append(tree)
                    formulas.append(formula)
                else:
                    tree = None

    nn_output = get_weights(tr_s, tr_l, trees, formulas)

    tr_vote, tr_rho = inc_evaluation(tr_s, tr_l, nn_output)
    if te_s is None:
        te_vote, te_rho = None, None
    else:
        te_vote, te_rho = inc_evaluation(te_s, te_l, nn_output)

    output_dict = {'formula': formulas, 'weight': nn_output, 'tr_vote': tr_vote,
                    'te_vote': te_vote, 'tr_rho': tr_rho, 'te_rho': te_rho}
    return output_dict

# ==============================================================================
# -- k-fold Learning() ---------------------------------------------------------
# ==============================================================================
def kfold_learning(signals, labels, timepoints, args):
    print('***************************************************************')
    print('(Number of signals, dimension, timepoints):', signals.shape)
    tic         = time.time()
    seed_value  = random.randrange(sys.maxsize)
    random.seed(seed_value)
    primitives  = make_stl_primitives1(signals)

    k_fold      = args.fold
    kf = KFold(n_splits = k_fold)
    formulas, weights   = (np.empty(k_fold, dtype = object) for i in range(2))
    tr_vote, te_vote    = (np.empty(k_fold, dtype = object) for i in range(2))
    tr_rho, te_rho      = (np.empty(k_fold, dtype = object) for i in range(2))

    for k, (tr_ind, te_ind) in enumerate(kf.split(signals)):
        tr_s, tr_l = signals[tr_ind], labels[tr_ind]
        te_s, te_l = signals[te_ind], labels[te_ind]
        print('***********************************************************')
        print("Fold {}:".format(k + 1))
        print('***********************************************************')
        rho_path   = [np.inf for signal in tr_s]
        res = boosted_trees(tr_s, tr_l, te_s, te_l, rho_path, primitives, args)
        formulas[k], weights[k] = res['formula'], res['weight']
        tr_vote[k], te_vote[k]  = res['tr_MCR_vote'], res['te_MCR_vote']
        tr_rho[k], te_rho[k]    = res['tr_MCR_rho'], res['te_MCR_rho']

    for k in range(k_fold):
        print('**********************************************************')
        print("Fold {}:".format(k + 1))
        print("Formula: {} \nWeights: {}".format(formulas[k], weights[k]))
        print("Train vote: {} \nTest vote: {}".format(tr_vote[k], te_vote[k]))
        print("Train rho: {} \nTest rho: {}".format(tr_rho[k], te_rho[k]))

    print("Seed:", seed_value)
    toc = time.time()
    print('Total Runtime:', toc - tic)

# ==============================================================================
# -- Parse Arguments() ---------------------------------------------------------
# ==============================================================================
def get_argparser():
    parser = argparse.ArgumentParser(formatter_class =
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--depth', metavar='D', type=int,
                        default = 1, help='maximum depth of the decision tree')
    parser.add_argument('-n', '--numtree', metavar='N', type=int,
                        default = 1, help='Number of decision trees')
    parser.add_argument('-k', '--fold', metavar='K', type=int,
                        default=2, help='K-fold cross-validation')
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
    args        = get_argparser().parse_args()
    file_name   = get_path(args.file)
    mat_data    = loadmat(file_name)
    timepoints  = mat_data['t'][0]
    labels      = mat_data['labels'][0]
    signals     = mat_data['data']

    kfold_learning(signals, labels, timepoints, args)
