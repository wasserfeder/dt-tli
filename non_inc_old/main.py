
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
from sklearn.model_selection import KFold
import random
import pickle
from non_inc_inf import Learning_Class
import sys
sys.path.append("/home/erfan/iitchs/catl_planning/python-stl/stl")
from stl import Trace


# ==============================================================================
# ---- Non-Incremental Main ----------------------------------------------
# ==============================================================================
class Non_Inc_Main(object):
    def __init__(self, args):
        self.args       = args
        self.filename   = get_path(self.args.file)
        self.mat_data   = loadmat(self.filename)
        self.timepoints = self.mat_data['t'][0]
        self.labels     = self.mat_data['labels'][0]
        self.signals    = self.mat_data['data']
        self.num_dimension = len(self.signals[0])
        self.varnames = ['x_{}'.format(i) for i in range(self.num_dimension)]
        self.traces = []
        for i in range(len(self.signals)):
            data = [self.signals[i][j] for j in range(self.num_dimension)]
            trace = Trace(self.varnames, self.timepoints, data)
            self.traces.append(trace)

        self.primitives1    = make_stl_primitives1(self.signals)
        self.primitives2    = make_stl_primitives2(self.signals)
        self.primitives     = self.primitives1
        self.M              = 100
        signals_shape   = self.signals.shape
        print('**********************************************************')
        print('(Number of signals, dimension, timepoints):', signals_shape)

        self.time_init  = time.time()
        self.kfold      = self.args.fold
        self.depth      = self.args.depth
        self.numtree    = self.args.numtree
        self.kfold_data = self.kfold_splitting()

        if self.args.action == 'learn':
            self.learning_results = self.kfold_learning()
        else:
            self.learning_results = self.kfold_cross_validation()


    ### method for splitting the signals based on the kfold cross validation
    def kfold_splitting(self):
        kfold_data = []
        if self.kfold <= 1:
            data_dict = {"tr_s":self.signals,"tr_t":self.traces,
                        "tr_l": self.labels, "te_s": None, "te_t": None,
                        "te_l": None}
            kfold_data.append(data_dict)
        else:
            kf = KFold(n_splits = self.kfold)
            for tr_index, te_index in kf.split(self.signals):
                tr_signals  = np.array([self.signals[i] for i in tr_index])
                tr_traces   = np.array([self.traces[i] for i in tr_index])
                tr_labels   = np.array([self.labels[i] for i in tr_index])
                te_signals  = np.array([self.signals[i] for i in te_index])
                te_traces   = np.array([self.traces[i] for i in te_index])
                te_labels   = np.array([self.labels[i] for i in te_index])
                data_dict = {"tr_s":tr_signals,"tr_t":tr_traces,
                            "tr_l":tr_labels,"te_s": te_signals,
                            "te_t": te_traces, "te_l": te_labels}
                kfold_data.append(data_dict)
        return kfold_data


    ### main method for the learning part
    def kfold_learning(self):
        learning_results = []
        for k in range(self.kfold):
            print('***********************************************************')
            print("Fold {}".format(k+1))
            print('***********************************************************')
            learning_dict = self.non_inc_learning(self.kfold_data[k])
            learning_results.append(learning_dict)

        return learning_results


    ### Learning method for the non-incremental framework
    def non_inc_learning(self, data):
        tr_s, tr_l = data["tr_s"], data["tr_l"]
        D_t = np.true_divide(np.ones(len(tr_s)), len(tr_s))
        rho_path = [np.inf for signal in tr_s]
        trees, formulas  = [None] * self.numtree, [None] * self.numtree
        weights, epsilon = [0] * self.numtree, [0] * self.numtree
        prunes = [None] * self.numtree

        learning = Learning_Class(self.primitives, self.args)   

        t = 0
        while t < self.numtree:
            if self.args.prune:
                root_info = learning.best_prim(data, rho_path, D_t)
                learning.prunes = []
                trees[t], prunes[t] = learning.pruned_tree(data, rho_path,
                                    root_info, self.depth, D_t)
            else:
                trees[t] = learning.normal_tree(data, rho_path, self.depth, D_t)
            rhos = [trees[t].tree_robustness(signal, np.inf) for signal in tr_s]  # # TODO: needs edit
            formulas[t] = trees[t].get_formula()
            pred_labels = np.array([trees[t].classify(signal) for signal in tr_s])
            for i in range(len(tr_s)):
                if tr_l[i] != pred_labels[i]:
                    epsilon[t] = epsilon[t] + D_t[i]

            if epsilon[t] > 0 and (epsilon[t] <= 0.5):
                weights[t] = 0.5 * np.log(1/epsilon[t] - 1)
            else:
                weights[t] = self.M

            if epsilon[t] <= 0.5:
                D_t = np.multiply(D_t, np.exp(np.multiply(-weights[t],
                                        np.multiply(tr_l, pred_labels))))
                D_t = np.true_divide(D_t, sum(D_t))
                t = t + 1

        tr_MCR = 100 * self.bdt_evaluation(data, trees, weights, 'train')
        if te_s is None:
            te_MCR = None
        else:
            te_MCR = 100 * self.bdt_evaluation(data, trees, weights, 'test')
        learning_dict = {'trees': trees, 'formulas': formulas,
            'weights': weights, 'tr_MCR': tr_MCR, 'te_MCR': te_MCR}

        return learning_dict




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
    print('Runtime:', dt)  ## # TODO: Edit the method to make it compatible with new formulations


# ==============================================================================
# -- Parse Arguments() ---------------------------------------------------------
# ==============================================================================
def get_argparser():
    parser = argparse.ArgumentParser(formatter_class =
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--action', metavar='A', default='learn', help=
                            """
                            'learn': builds a classifier
                            'cv': performs cross validation
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
    non_inc_class = Non_Inc_Main(args)

    dt = time.time() - non_inc_class.time_init
    print("Runtime:", dt)
