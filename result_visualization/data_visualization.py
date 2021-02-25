## Imports
import numpy as np
import math
from scipy.io import loadmat
import argparse
from os import path
import os
import matplotlib.pyplot as plt
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.weight'] = 'black'
# plt.rcParams['font.size'] = '18'


def draw_results(mat_data):
    timepoints      = mat_data['t'][0]
    labels          = mat_data['labels'][0]
    signals         = mat_data['data']
    num_signals     = len(signals)
    num_dimensions  = len(signals[0])
    pos_indices, neg_indices = [], []
    for i in range(len(labels)):
        if labels[i] >= 0:
            pos_indices.append(i)
        else:
            neg_indices.append(i)

    datafig, axs = plt.subplots(num_dimensions, 2)
    for j in range(num_dimensions):
        for i in pos_indices:
            if num_dimensions > 1:
                axs[j][0].plot(timepoints, signals[i][j])
            else:
                axs[0].plot(timepoints, signals[i][j])
        if num_dimensions > 1:
            axs[j, 0].grid(b = True, which = 'major')
        else:
            axs[0].grid(True)
    if num_dimensions > 1:
        axs[0, 0].set_title("Positive Signals")
        axs[0, 0].set_ylabel('X (m)')
        axs[1, 0].set_ylabel('Y (m)')
        axs[1,0].set_xlabel('Time')
    else:
        axs[0].set_title("Positive Signals")
        axs[0].set_ylabel('Signal value')
        axs[0].set_xlabel('Time')


    for j in range(num_dimensions):
        for i in neg_indices:
            if num_dimensions > 1:
                axs[j, 1].plot(timepoints, signals[i][j])
            else:
                axs[1].plot(timepoints, signals[i][j])
        if num_dimensions > 1:
            axs[j, 1].grid(b = True, which = 'major')
        else:
            axs[1].grid(True)
    if num_dimensions > 1:
        axs[0, 1].set_title("Negative Signals")
        axs[1,1].set_xlabel('Time')
    else:
        axs[1].set_title("Negative Signals")
        axs[1].set_xlabel('Time')


    datafig.savefig('/home/erfan/Documents/University/Projects/Learning Specifications/dt-tli/Figures/data_time.eps', format='eps')
    datafig.savefig('/home/erfan/Documents/University/Projects/Learning Specifications/dt-tli/Figures/data_time.png', format='png')
    plt.show()


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
