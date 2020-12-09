## Imports
import numpy as np
import math
from scipy.io import loadmat
import argparse
from os import path
import os
import matplotlib.pyplot as plt


def draw_results(mat_data):
    timepoints  = list(mat_data['t'][0])
    labels      = list(mat_data['labels'][0])
    num_signals = len(labels)
    data        = [mat_data['data'][i] for i in range(num_signals)]
    dimension   = len(data[0])
    pos_indices, neg_indices = [], []
    for i in range(len(labels)):
        if labels[i] >= 0:
            pos_indices.append(i)
        else:
            neg_indices.append(i)

    # print(dimension)
    # print(data)


    datafig, axs = plt.subplots(dimension, 2)
    for j in range(dimension):
        for i in pos_indices:
            if dimension > 1:
                axs[j][0].plot(timepoints, data[i][j])
            else:
                axs[0].plot(timepoints, data[i][j])
        if dimension > 1:
            axs[j, 0].grid(True)
        else:
            axs[0].grid(True)
    if dimension > 1:
        axs[0, 0].set_title("Positive Signals")
    else:
        axs[0].set_title("Positive Signals")


    for j in range(dimension):
        for i in neg_indices:
            if dimension > 1:
                axs[j, 1].plot(timepoints, data[i][j])
            else:
                axs[1].plot(timepoints, data[i][j])
        if dimension > 1:
            axs[j, 1].grid(True)
        else:
            axs[1].grid(True)
    if dimension > 1:
        axs[0, 1].set_title("Negative Signals")
    else:
        axs[1].set_title("Negative Signals")

    plt.show(block = True)
    datafig.savefig('/Users/erfanaasi/Documents/University/Projects/Learning Specifications/dt-tli/Results/data_time.eps', format='eps')

def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file', help='.mat file containing the data')
    return parser


def get_path(f):
    return path.join(os.getcwd(), f)


if __name__ == '__main__':
    args        = get_argparser().parse_args()
    mat_file    = get_path(args.file)
    mat_data    = loadmat(mat_file)
    draw_results(mat_data)
