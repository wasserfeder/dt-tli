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
import pickle
from tree_visualization import draw_tree
import matplotlib.pyplot as plt
from non_inc_inf import Non_Inc_Build_Tree

# ==============================================================================
# -- The main class for learning the formulas and the predictors ---------------
# ==============================================================================

class Learning_Main(object):
    def __init__(self, filename, args):
        self.args           = args
        self.mat_data       = loadmat(filename)
        self.timepoints     = self.mat_data['t'][0]
        self.labels         = self.mat_data['labels'][0]
        self.signals        = self.mat_data['data']
        self.num_dimensions = len(self.signals[0])
        self.primitives1    = make_stl_primitives1(self.signals)
        self.primitives2    = make_stl_primitives2(self.signals)
        self.primitives     = self.primitives1
        self.M              = 100
        signals_shape       = self.signals.shape
        print('**************************************************************')
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
            data_dict = {"tr_signals": self.signals, "tr_labels": self.labels,
                        "te_signals": None, "te_labels": None}
            kfold_data.append(data_dict)
        else:
            kf = KFold(n_splits = self.kfold)
            for tr_index, te_index in kf.split(self.signals):
                tr_signals  = np.array([self.signals[i] for i in tr_index])
                tr_labels   = np.array([self.labels[i] for i in tr_index])
                te_signals  = np.array([self.signals[i] for i in te_index])
                te_labels   = np.array([self.labels[i] for i in te_index])
                data_dict = {"tr_signals": tr_signals, "tr_labels": tr_labels,
                            "te_signals": te_signals, "te_labels": te_labels}
                kfold_data.append(data_dict)
        return kfold_data


### main method for the learning part
    def kfold_learning(self):
        learning_results = []
        for k in range(self.kfold):
            print('***************************************************************')
            print("Fold {}".format(k+1))
            print('***************************************************************')
            if not self.args.inc:
                learning_dict = self.non_inc_learning(self.kfold_data[k])
            else:
                learning_dict = self.inc_learning(self.kfold_data[k])
            learning_results.append(learning_dict)

        return learning_results


### Learning method for the non-incremental framework
    def non_inc_learning(self, data_dict):
        tr_s, tr_l = data_dict["tr_signals"], data_dict["tr_labels"]
        te_s, te_l = data_dict["te_signals"], data_dict["te_labels"]
        D_t = np.true_divide(np.ones(len(tr_s)), len(tr_s))
        rho_path = [np.inf for signal in tr_s]
        trees, formulas  = [None] * self.numtree, [None] * self.numtree
        weights, epsilon = [0] * self.numtree, [0] * self.numtree

        t = 0
        while t < self.numtree:
            trees[t] = build_tree(tr_s, tr_l, rho_path, self.depth, self.primitives, D_t, self.args, None)
            rhos = [trees[t].tree_robustness(signal, np.inf) for signal in tr_s]
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
            # else:
            #     trees = trees[:-1]
            #     formulas = formulas[:-1]
            #     weights = weights[:-1]
            #     epsilon = epsilon[:-1]


        if not args.inc_eval:
            tr_MCR = 100 * self.bdt_evaluation(tr_s, tr_l, trees, weights)
            if te_s is None:
                te_MCR = None
            else:
                te_MCR = 100 * self.bdt_evaluation(te_s, te_l, trees, weights)
            learning_dict = {'trees': trees, 'formulas': formulas,
                'weights': weights, 'tr_MCR': tr_MCR, 'te_MCR': te_MCR,
                'tr_MCR_vector': None, 'te_MCR_vector': None}

        else:
            tr_MCR_vector, tr_MCR = inc_normal_evaluation(tr_s, tr_l, trees,
                                                            weights, numtree)
            if te_s is None:
                te_MCR = None
            else:
                te_MCR_vector, te_MCR = inc_normal_evaluation(te_s, te_l, trees,
                                                            weights, numtree)
            learning_dict = {'trees': trees, 'formulas': formulas,
                'weights': weights, 'tr_MCR': tr_MCR, 'te_MCR': te_MCR,
                'tr_MCR_vector': tr_MCR_vector, 'te_MCR_vector': te_MCR_vector}
        return learning_dict




    def inc_learning(self, data_dict):
        return None, None, None, None, None, None, None



    def kfold_cross_validation(self):
        return None



    def bdt_evaluation(self, signals, labels, trees, weights):
        test = np.zeros(len(signals))
        for i in range(self.numtree):
            test = test + weights[i] * np.array([trees[i].classify(signal)
                                                        for signal in signals])

        test = np.sign(test)
        return np.count_nonzero(labels - test) / float(len(labels))



    def show_results(self):
        for k in range(self.kfold):
            print("Fold {}:".format(k + 1))
            print("Formula:", self.learning_results[k]['formulas'])
            print("Parent:", self.learning_results[k]['trees'][0].parent)
            print("left Parent:", self.learning_results[k]['trees'][0].left.left.parent.primitive)
            print("right Parent:", self.learning_results[k]['trees'][0].left.right.parent.primitive)
            print("Weight(s):", self.learning_results[k]['weights'])
            print("Train MCR:", self.learning_results[k]['tr_MCR'])
            print("Test MCR:", self.learning_results[k]['te_MCR'])
            print("Train MCR Vector:",self.learning_results[k]['tr_MCR_vector'])
            print("Test MCR Vector:", self.learning_results[k]['te_MCR_vector'])
            self.draw_results(k)


    def draw_results(self, k):
        data_dict = self.kfold_data[k]
        tr_s, tr_l = data_dict["tr_signals"], data_dict["tr_labels"]
        te_s, te_l = data_dict["te_signals"], data_dict["te_labels"]
        trainingfig, tr_axs = plt.subplots(self.num_dimensions, 2)
        for j in range(self.num_dimensions):
            for i in range(len(tr_s)):
                if tr_l[i] > 0:
                    if self.num_dimensions > 1:
                        tr_axs[j, 0].plot(self.timepoints, tr_s[i][j])
                        tr_axs[j, 0].grid(b = True, which = 'major')
                    else:
                        tr_axs[0].plot(self.timepoints, tr_s[i][j])
                        tr_axs[0].grid(True)
                else:
                    if self.num_dimensions > 1:
                        tr_axs[j, 1].plot(self.timepoints, tr_s[i][j])
                        tr_axs[j, 1].grid(b = True, which = 'major')
                    else:
                        tr_axs[1].plot(self.timepoints, tr_s[i][j])
                        tr_axs[1].grid(True)
        if self.num_dimensions > 1:
            tr_axs[0, 0].set_title("Positive Signals, Fold {}".format(k+1))
            tr_axs[0, 1].set_title("Negative Signals, Fold {}".format(k+1))
            tr_axs[self.num_dimensions-1, 0].set_xlabel("Time")
            tr_axs[self.num_dimensions-1, 1].set_xlabel("Time")
        else:
            tr_axs[0].set_title("Positive Training Signals, Fold {}".format(k+1))
            tr_axs[1].set_title("Negative Training Signals, Fold {}".format(k+1))
            tr_axs[0].set_xlabel("Time")
            tr_axs[1].set_xlabel("Time")

        testfig, te_axs = plt.subplots(self.num_dimensions, 2)
        for j in range(self.num_dimensions):
            for i in range(len(te_s)):
                if te_l[i] > 0:
                    if self.num_dimensions > 1:
                        te_axs[j, 0].plot(self.timepoints, te_s[i][j])
                        te_axs[j, 0].grid(b = True, which = 'major')
                    else:
                        te_axs[0].plot(self.timepoints, te_s[i][j])
                        te_axs[0].grid(True)
                else:
                    if self.num_dimensions > 1:
                        te_axs[j, 1].plot(self.timepoints, te_s[i][j])
                        te_axs[j, 1].grid(b = True, which = 'major')
                    else:
                        te_axs[1].plot(self.timepoints, te_s[i][j])
                        te_axs[1].grid(True)
        if self.num_dimensions > 1:
            te_axs[0, 0].set_title("Positive Test Signals, Fold {}".format(k+1))
            te_axs[0, 1].set_title("Negative Test Signals, Fold {}".format(k+1))
            te_axs[self.num_dimensions-1, 0].set_xlabel("Time")
            te_axs[self.num_dimensions-1, 1].set_xlabel("Time")
        else:
            te_axs[0].set_title("Positive Test Signals, Fold {}".format(k+1))
            te_axs[1].set_title("Negative Test Signals, Fold {}".format(k+1))
            te_axs[0].set_xlabel("Time")
            te_axs[1].set_xlabel("Time")

        trainingfig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/incremental_predictor/dt-tli/Figures/training_signals_fold{}.eps'.format(k+1), format='eps')
        trainingfig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/incremental_predictor/dt-tli/Figures/training_signals_fold{}.png'.format(k+1), format='png')
        testfig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/incremental_predictor/dt-tli/Figures/test_signals_fold{}.eps'.format(k+1), format='eps')
        testfig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/incremental_predictor/dt-tli/Figures/test_signals_fold{}.png'.format(k+1), format='png')

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
    parser.add_argument('-ie', '--inc_eval', metavar='IE', type=int,
                        default=0, help='Incremental or Non-incremental Evaluation')
    parser.add_argument('-k', '--fold', metavar='K', type=int,
                        default=0, help='K-fold cross-validation')
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
    learning_main = Learning_Main(get_path(args.file), args)
    learning_main.show_results()

    dt = time.time() - learning_main.time_init
    print("Runtime:", dt)
