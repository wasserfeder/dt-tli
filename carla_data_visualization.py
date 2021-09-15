import numpy as np
import math
from scipy.io import loadmat, savemat
import argparse
from os import path
import os
import matplotlib.pyplot as plt
import copy

plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
plt.rcParams['text.usetex'] = True
plt.rcParams['font.weight'] = 'black'
plt.rcParams['font.size'] = '30'




def draw_results(mat_data):
    timepoints      = np.array(mat_data['t'][0])
    labels          = mat_data['labels'][0]
    signals         = mat_data['data']
    num_signals     = len(signals)
    num_dimensions  = len(signals[0])


    carla_yz_fig = plt.figure()
    for i in range(num_signals):
        if labels[i] > 0:
            plt.plot(signals[i][0], signals[i][1], color='green', linewidth = 1, label=r'\textbf{Positive Signals}')
        else:
            plt.plot(signals[i][0], signals[i][1], color='red', linewidth = 1, label=r'\textbf{Negative Signals}')

    # plt.legend(loc = 'lower right')
    plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
    # plt.plot([0, 80], [21.31, 21.31], color = 'black', linewidth = 6)
    # plt.plot([11.10, 11.10], [15, 45], color = 'black', linewidth = 6)
    # plt.plot([30.85, 30.85], [15, 45], color = 'black', linewidth = 6, linestyle = '--')
    # plt.xlim((0, 80))
    # plt.ylim((15, 45))
    plt.xlabel(r'\textbf{Relative Y (m)}')
    plt.ylabel(r'\textbf{Relative Z (m)}')
    plt.grid(True)
    plt.show(block = False)
    # plt.title(r'\textbf{Carla Scenario}')
    # carla_yz_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/carla_yz.eps', format='eps')
    # carla_yz_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/carla_yz.png', format='png')



    carla_vyvz_fig = plt.figure()
    for i in range(num_signals):
        if labels[i] > 0:
            plt.plot(signals[i][2], signals[i][3], color='green', linewidth = 1, label=r'\textbf{Positive Signals}')
        else:
            plt.plot(signals[i][2], signals[i][3], color='red', linewidth = 1, label=r'\textbf{Negative Signals}')

    # plt.legend(loc = 'lower right')
    plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
    # plt.plot([0, 80], [21.31, 21.31], color = 'black', linewidth = 6)
    # plt.plot([11.10, 11.10], [15, 45], color = 'black', linewidth = 6)
    # plt.plot([30.85, 30.85], [15, 45], color = 'black', linewidth = 6, linestyle = '--')
    # plt.xlim((0, 80))
    # plt.ylim((15, 45))
    plt.xlabel(r'\textbf{Relative V-Y (m/s)}')
    plt.ylabel(r'\textbf{Relative V-Z (m/s)}')
    plt.grid(True)
    plt.show(block = False)
    # plt.title(r'\textbf{Carla Scenario}')
    # carla_vyvz_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/carla_vyvz.eps', format='eps')
    # carla_vyvz_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/carla_vyvz.png', format='png')



    carla_y_fig = plt.figure()
    for i in range(num_signals):
        if labels[i] > 0:
            plt.plot(timepoints, signals[i][0], color = 'green', linewidth = 1, label=r'\textbf{Positive Signals}')
        else:
            plt.plot(timepoints, signals[i][0], color = 'red', linewidth = 1, label=r'\textbf{Negative Signals}')
    # plt.legend(loc = 'upper right')
    left, right = [370, 14.01], [485, 14.01]
    plt.plot([left[0], right[0]], [left[1], right[1]], color = 'black', linewidth = 6)
    plt.plot([left[0], left[0]], [0, left[1]], color = 'black', linewidth=2, linestyle = '--')
    plt.plot([right[0], right[0]], [0, right[1]], color = 'black', linewidth=2, linestyle = '--')
    plt.text(375, 3, r'\textbf{t=370}')
    plt.text(465, 3, r'\textbf{t=485}')
    plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
    plt.xlim((300, 500))
    plt.ylim((0, 30))
    plt.xlabel(r'\textbf{t (s)}')
    plt.ylabel(r'\textbf{Y (m)}')
    plt.grid(True)
    plt.show(block = False)
    # plt.title(r'\textbf{X component of naval signals}')
    # naval_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval_x.eps', format='eps')
    # naval_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval_x.png', format='png')



    carla_z_fig = plt.figure()
    for i in range(num_signals):
        if labels[i] > 0:
            plt.plot(timepoints, signals[i][1], color = 'green', linewidth = 1, label=r'\textbf{Positive Signals}')
        else:
            plt.plot(timepoints, signals[i][1], color = 'red', linewidth = 1, label=r'\textbf{Negative Signals}')
    # plt.legend(loc = 'upper right')
    plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
    plt.xlim((300, 500))
    plt.ylim((0, 4))
    plt.xlabel(r'\textbf{t (s)}')
    plt.ylabel(r'\textbf{Z (m)}')
    plt.grid(True)
    plt.show(block = False)
    # plt.title(r'\textbf{X component of naval signals}')
    # naval_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval_x.eps', format='eps')
    # naval_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval_x.png', format='png')




    carla_vy_fig = plt.figure()
    for i in range(num_signals):
        if labels[i] > 0:
            plt.plot(timepoints, signals[i][2], color = 'green', linewidth = 1, label=r'\textbf{Positive Signals}')
        else:
            plt.plot(timepoints, signals[i][2], color = 'red', linewidth = 1, label=r'\textbf{Negative Signals}')
    # plt.legend(loc = 'upper right')
    left, right = [370, 7.45], [485, 7.45]
    plt.plot([left[0], right[0]], [left[1], right[1]], color = 'black', linewidth = 6)
    plt.plot([left[0], left[0]], [0, left[1]], color = 'black', linewidth=2, linestyle = '--')
    plt.plot([right[0], right[0]], [0, right[1]], color = 'black', linewidth=2, linestyle = '--')
    plt.text(375, 3, r'\textbf{t=370}')
    plt.text(465, 3, r'\textbf{t=485}')
    plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
    plt.xlim((300, 500))
    plt.ylim((0, 15))
    plt.xlabel(r'\textbf{t (s)}')
    plt.ylabel(r'\textbf{v_y (m/s)}')
    plt.grid(True)
    plt.show(block = False)
    # plt.title(r'\textbf{X component of naval signals}')
    # naval_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval_x.eps', format='eps')
    # naval_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval_x.png', format='png')


    carla_vz_fig = plt.figure()
    for i in range(num_signals):
        if labels[i] > 0:
            plt.plot(timepoints, signals[i][3], color = 'green', linewidth = 1, label=r'\textbf{Positive Signals}')
        else:
            plt.plot(timepoints, signals[i][3], color = 'red', linewidth = 1, label=r'\textbf{Negative Signals}')
    # plt.legend(loc = 'upper right')
    plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
    plt.xlim((300, 500))
    plt.ylim((0, 3))
    plt.xlabel(r'\textbf{t (s)}')
    plt.ylabel(r'\textbf{v_z (m/s)}')
    plt.grid(True)
    plt.show(block = False)
    # plt.title(r'\textbf{X component of naval signals}')
    # naval_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval_x.eps', format='eps')
    # naval_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval_x.png', format='png')


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
