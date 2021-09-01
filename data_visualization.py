import numpy as np
import math
from scipy.io import loadmat, savemat
import argparse
from os import path
import os
import matplotlib.pyplot as plt
import copy




def draw_results(mat_data):
    timepoints      = mat_data['t'][0]
    labels          = mat_data['labels'][0]
    signals         = mat_data['data']
    num_signals     = len(signals)
    num_dimensions  = len(signals[0])
    datafig, axs = plt.subplots(1, 1)
    for i in range(num_signals):
        axs.plot(signals[i][0], signals[i][1])


    # rotated_signals = rotate_signals(signals)
    # data_dict = {'data':rotated_signals, 't':timepoints, 'labels':labels}
    # savemat("rotated_naval.mat",data_dict)
    # newdatafig, newaxs = plt.subplots(1,1)
    # for i in range(num_signals):
    #     newaxs.plot(rotated_signals[i][0], rotated_signals[i][1])
    #

    plt.show()

def rotate_signals(signals):
    num_signals = len(signals)
    num_dimensions = len(signals[0])
    timepoints = len(signals[0][0])

    new_signals = copy.deepcopy(signals)
    for i in range(num_signals):
        cx, cy = 0, 0
        for t in range(timepoints):
            new_signals[i][0][t] = ((signals[i][0][t] - cx) * math.cos(-np.pi/4) + (signals[i][1][t] - cy) * math.sin(-np.pi/4)) + cx
            new_signals[i][1][t] = (-(signals[i][0][t] - cx) * math.sin(-np.pi/4) + (signals[i][1][t] - cy) * math.cos(-np.pi/4)) + cy

    return new_signals






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
