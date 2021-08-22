
import numpy as np
import math
from scipy.io import loadmat, savemat
import argparse
from os import path
import os
from pso import pso_costFunc, PSO
import sys
sys.path.append("/home/erfan/iitchs/catl_planning/python-stl/stl")
from stl import Trace
from stl_prim import make_stl_primitives1, make_stl_primitives2



def get_indices(primitive, indices): # Done
    if primitive.op == 6 or primitive.op == 7:
        indices = get_indices(primitive.child, indices)
        return indices
    elif primitive.op == 3:
        for i in range(len(primitive.children)):
            indices.append(get_indices(primitive.children[i], indices))
        return indices
    elif primitive.op == 8:
        index = int(primitive.variable.split("_")[1])
        return index
    elif primitive.op == 5:
        indices = get_indices(primitive.left, indices)
        indices = get_indices(primitive.right, indices)
        return indices


def get_bounds(signals, signal_indices): # Done
    bounds = []
    for k in signal_indices:
        min_signals = []
        max_signals = []
        for i in range(len(signals)):
            min_signals.append(min(signals[i][k]))
            max_signals.append(max(signals[i][k]))
        min_pi = min(min_signals)
        max_pi = max(max_signals)
        bounds.append(min_pi)
        bounds.append(max_pi)
    max_t = len(signals[0][0]) - 1
    bounds.append(max_t)
    return bounds


def run_pso_optimization(signals, traces, labels, rho_path, primitive, D_t, args):
    signal_indices = get_indices(primitive, [])
    if isinstance(signal_indices, int):
        signal_indices = [signal_indices]
    bounds = get_bounds(signals, signal_indices)
    particle_swarm = PSO(signals, traces, labels, bounds, primitive, args)
    params, impurity = particle_swarm.optimize_swarm(rho_path, D_t)
    return params, impurity




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
    return path.join(os.getcwd(), f)



def main():
    args                = get_argparser().parse_args()
    filename            = get_path(args.file)
    mat_data            = loadmat(filename)
    timepoints          = mat_data['t'][0]
    labels              = mat_data['labels'][0]
    signals             = mat_data['data']
    D_t                 = np.true_divide(np.ones(len(signals)), len(signals))
    num_dimension       = len(signals[0])
    traces = []
    varnames = ['x_{}'.format(i) for i in range(num_dimension)]
    for i in range(len(signals)):
        data = [signals[i][j] for j in range(num_dimension)]
        trace = Trace(varnames, timepoints, data)
        traces.append(trace)

    print('(Number of signals, dimension, timepoints):', signals.shape)

    rho_path = [np.inf for signal in signals]
    primitives    = make_stl_primitives1(signals)
    primitive = primitives[0]
    best_params, impurity   = run_pso_optimization(signals, traces, labels, rho_path, primitive, D_t, args)
    print("best primitive:", best_params)
    print("impurity:", impurity)


if __name__ == '__main__':
    main()
