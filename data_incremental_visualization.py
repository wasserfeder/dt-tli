
import numpy as np
import math
from scipy.io import loadmat, savemat
import os
import sys
import argparse
import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
plt.rcParams['text.usetex'] = True
plt.rcParams['font.weight'] = 'black'
plt.rcParams['font.size'] = '22'



def signal_distance(pos_signal, neg_signal):
    distance = 0
    for j in range(len(pos_signal)):
        distance += (pos_signal[j] - neg_signal[j])**2

    return np.sqrt(distance)



def compute_distance(pos_signals, neg_signals, timepoints):
    distances = np.zeros(len(timepoints))
    min_distances = np.zeros(len(timepoints))
    max_distances = np.zeros(len(timepoints))
    mean_distances = np.zeros(len(timepoints))
    std_distances = np.zeros(len(timepoints))
    len_pos, len_neg = len(pos_signals), len(neg_signals)
    for t in range(len(timepoints)):
        pos_signals_par = pos_signals[:, :, t]
        neg_signals_par = neg_signals[:, :, t]
        distance_array = []
        for p in range(len_pos):
            pos_signal = pos_signals_par[p]
            for n in range(len_neg):
                neg_signal = neg_signals_par[n]
                distance = signal_distance(pos_signal, neg_signal)
                distance_array.append(distance)
                distances[t] += distance
        distance_array = np.array(distance_array)
        max_distances[t] = np.max(distance_array)
        min_distances[t] = np.min(distance_array)
        mean_distances[t] = np.mean(distance_array)
        std_distances[t] = np.std(distance_array)

    return distances, max_distances, min_distances, mean_distances, std_distances



def get_argparser():
    parser = argparse.ArgumentParser(formatter_class =
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file', help='.mat file containing the data')
    return parser



def get_path(f):
    return os.path.join(os.getcwd(), f)



if __name__ == '__main__':
    args        = get_argparser().parse_args()
    file_name   = get_path(args.file)
    mat_data    = loadmat(file_name)
    timepoints  = mat_data['t'][0]
    labels      = mat_data['labels'][0]
    signals     = mat_data['data']

    pos_ind, neg_ind = np.where(labels > 0)[0], np.where(labels <= 0)[0]
    pos_signals, neg_signals = signals[pos_ind], signals[neg_ind]
    sum_d, max_d, min_d, mean_d, std_d = compute_distance(pos_signals, neg_signals, timepoints)

    distance_fig = plt.figure()
    plt.plot(timepoints/5, sum_d, linewidth = 4)
    plt.xlabel(r'\textbf{Time}')
    plt.ylabel(r'\textbf{Sum of Distance}')

    max_distance_fig = plt.figure()
    plt.plot(timepoints/5, max_d, linewidth = 4)
    plt.xlabel(r'\textbf{Time}')
    plt.ylabel(r'\textbf{Maximum Distance}')

    min_distance_fig = plt.figure()
    plt.plot(timepoints/5, min_d, linewidth = 4)
    plt.xlabel(r'\textbf{Time}')
    plt.ylabel(r'\textbf{Minimum Distance}')


    mean_distance_fig = plt.figure()
    plt.plot(timepoints/5, mean_d, linewidth = 4)
    plt.xlabel(r'\textbf{Time}')
    plt.ylabel(r'\textbf{Mean Distance}')


    std_distance_fig = plt.figure()
    plt.plot(timepoints/5, std_d, linewidth = 4)
    plt.xlabel(r'\textbf{Time}')
    plt.ylabel(r'\textbf{STD Distance}')


    first_deriv_fig = plt.figure()
    plt.plot(timepoints/5, np.gradient(mean_d), linewidth = 4)
    plt.xlabel(r'\textbf{Time}')
    plt.ylabel(r'\textbf{First Derivative}')


    plt.show()
