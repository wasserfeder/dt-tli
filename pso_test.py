
import numpy as np
import math
from scipy.io import loadmat, savemat
import argparse
from os import path
import os
from pso import pso_costFunc, PSO


def get_bounds(signals, signal_dimension=0):
    k = signal_dimension
    min_signals = []
    max_signals = []
    for i in range(len(signals)):
        min_signals.append(min(signals[i][k]))
        max_signals.append(max(signals[i][k]))
    min_pi = min(min_signals)
    max_pi = max(max_signals)
    max_t = len(signals[0][0]) - 1
    bounds = [min_pi, max_pi, max_t]
    return bounds



def run_pso_optimization(signals, labels, rho_path, signal_dimension=0):
    bounds = get_bounds(signals, signal_dimension)
    particle_swarm = PSO(signals, labels, bounds, signal_dimension)
    params, impurity = particle_swarm.optimize_swarm(rho_path)
    return params, impurity




def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

    print(signals.shape)
    print('Number of signals:', len(signals))
    print('Time points:', len(timepoints))

    rho_path = [np.inf for signal in signals]
    best_params, impurity   = run_pso_optimization(signals, labels, rho_path, signal_dimension = 1)
    print("best primitive:", best_params)
    print("impurity:", impurity)


if __name__ == '__main__':
    main()
