
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

def learn_formula(filename, depth, numtree, inc):
    ######### DEBUGGING Optimization Problem ################
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
    rho_path    = [np.inf for signal in signals]
    dt = time.time() - t0
    print('Setup time:', dt)


    t0 = time.time()
    tree = build_tree(signals, labels, depth, primitives1, rho_path)
    formula = tree.get_formula()
    print('Formula:', formula)

    missclassification_rate = 100 * evaluation(signals, labels, tree)
    print("Misclassification Rate:", missclassification_rate)

    dt = time.time() - t0
    print('Runtime:', dt)

##########
    # datafig, axs = plt.subplots(2, 2)
    # x_values = [85, 300]
    # y_values = [35.32, 35.32]
    # x_values1 = [85, 85]
    # y_values1 = [0, 35.32]
    # counter = 0
    # for i in sat_indices:
    #     if labels[i] > 0:
    #         axs[0][0].plot(timepoints, signals[i][1])
    #     else:
    #         axs[0][1].plot(timepoints, signals[i][1])
    #         counter = counter + 1
    # axs[0][0].set_title("Correctly classified")
    # axs[0][0].grid(True)
    # axs[0][1].grid(True)
    # axs[0, 0].set_ylabel('Signal value')
    # axs[0][1].set_title('Misclassified Signals')
    # for i in unsat_indices:
    #     if labels[i] < 0:
    #         axs[1][0].plot(timepoints, signals[i][1])
    #     else:
    #         axs[1][1].plot(timepoints, signals[i][1])
    #         counter = counter + 1
    # axs[1][0].set_ylabel('Signal value')
    # axs[1][0].set_xlabel('Time')
    # axs[1][0].grid(True)
    # axs[1][1].grid(True)
    # axs[1,1].set_xlabel('Time')
    # p1, p2 = [85,35.32], [300, 35.32]
    # x_values = [85, 300]
    # y_values = [35.32, 35.32]
    # axs[0][0].plot(x_values, y_values, color = 'red', linestyle = '--', linewidth = 5)
    # axs[0][0].plot(x_values1, y_values1, color = 'red', linestyle = '--', linewidth = 5)
    # axs[0][1].plot(x_values, y_values, color = 'red', linestyle = '--', linewidth = 5)
    # axs[0][1].plot(x_values1, y_values1, color = 'red', linestyle = '--', linewidth = 5)
    # axs[1][0].plot(x_values, y_values, color = 'red', linestyle = '--', linewidth = 5)
    # axs[1][0].plot(x_values1, y_values1, color = 'red', linestyle = '--', linewidth = 5)
    # axs[1][1].plot(x_values, y_values, color = 'red', linestyle = '--', linewidth = 5)
    # axs[1][1].plot(x_values1, y_values1, color = 'red', linestyle = '--', linewidth = 5)
    # plt.show(block = True)
    # datafig.savefig('/home/erfan/Documents/University/Projects/Learning Specifications/dt-tli/Figures/misclassified.eps', format='eps')
    # datafig.savefig('/home/erfan/Documents/University/Projects/Learning Specifications/dt-tli/Figures/misclassified.png', format='png')
    # print("Other Misclassification Rate:", 100 * counter / float(len(labels)))



    # trees, formulas = [None] * numtree, [None] * numtree        # for boosted
    # weights, epsilon = [0] * numtree, [0] * numtree             # for boosted
    # D_t = np.true_divide(np.ones(len(signals)), len(signals))   # for boosted
    # intial_rho = None
    # rho_max = 2
    # stop_condition = [perfect_stop, depth_stop]
    # models = [SimpleModel(signal) for signal in traces.signals]

    # for t in range(numtree):
    #     trees[t] = learn_stlinf(signals, pdist, depth, inc, initial_rho,
    #                             stop_condition, verbose)
    #     formulas[t] = trees[t].get_formula()
    #     robustnesses = [robustness(formulas[t], model) for model in models]     #### TODO
    #     robust_err = 1 - np.true_divide(np.abs(robustnesses), rho_max)
    #     epsilon[t] = sum(np.multiply(pdist, robust_err))
    #     weights[t] = 0.5 * np.log(1/epsilon[t] - 1)
    #     pred_labels = [trees[t].classify(s) for s in traces.signals] ##### TODO
    #     pdist = np.multiply(pdist, np.exp(np.multiply(-weights[t],
    #                                         np.multiply(labels, pred_labels))))
    #     pdist = np.true_divide(pdist, sum(pdist))

    # ###### Evaluation
    # test = np.zeros(numtree)
    # for i in range(numtree):
    #     test = test + np.multiply(weights[i], np.array([trees[i].classify(s) for s in data]))

    # test = np.sign(test)
    # # test = np.array([np.sign(sum(np.multiply(weights, trees.classify(s))))
    # #                                                             for s in data])

    # print(np.count_nonzero(labels - test) / float(len(labels)))


def evaluation(signals, labels, tree):
    labels = np.array(labels)
    predictions = np.array([tree.classify(signal) for signal in signals])
    return np.count_nonzero(labels-predictions)/float(len(labels))



# ==============================================================================
# -- Parse Arguments() ---------------------------------------------------------
# ==============================================================================

def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    learn_formula(get_path(args.file), args.depth, args.numtree, args.inc)
