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
    timepoints      = np.array(mat_data['t'][0])/5
    labels          = mat_data['labels'][0]
    signals         = mat_data['data']
    num_signals     = len(signals)
    num_dimensions  = len(signals[0])


    navalfig = plt.figure()
    pos_counter, neg_counter = 0, 0
    for i in range(num_signals):
        if labels[i] > 0 and pos_counter <= 400:
            pos_counter += 1
            plt.plot(signals[i][0], signals[i][1], color='green', linewidth = 1, label=r'\textbf{Positive Signals}')
        elif labels[i] < 0 and neg_counter <= 400:
            neg_counter += 1
            plt.plot(signals[i][0], signals[i][1], color='red', linewidth = 1, label=r'\textbf{Negative Signals}')

    # plt.legend(loc = 'lower right')
    plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
    plt.plot([0, 80], [21.31, 21.31], color = 'black', linewidth = 6)
    plt.plot([11.10, 11.10], [15, 45], color = 'black', linewidth = 6)
    plt.plot([30.85, 30.85], [15, 45], color = 'black', linewidth = 6, linestyle = '--')
    plt.xlim((0, 80))
    plt.ylim((15, 45))
    plt.xlabel(r'\textbf{X (m)}')
    plt.ylabel(r'\textbf{Y (m)}')
    plt.grid(True)
    plt.show(block = False)
    # plt.title(r'\textbf{Original Naval Scenario}')
    # navalfig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval.eps', format='eps')
    # navalfig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval.png', format='png')



    naval_x_fig = plt.figure()
    pos_counter, neg_counter = 0, 0
    for i in range(num_signals):
        if labels[i] > 0 and pos_counter <= 400:
            pos_counter += 1
            plt.plot(timepoints[:31], signals[i][0][:31], color='green', linewidth = 1, label=r'\textbf{Positive Signals}')
        elif labels[i] < 0 and neg_counter <= 400:
            neg_counter +=1
            plt.plot(timepoints[:31], signals[i][0][:31], color='red', linewidth = 1, label=r'\textbf{Negative Signals}')
    # plt.legend(loc = 'upper right')
    left_bottom, right_bottom = [15, 40], [25, 40]
    left_top, right_top = [15, 47], [25, 47]
    plt.plot([left_bottom[0], right_bottom[0]], [left_bottom[1], right_bottom[1]], color = 'black', linewidth = 6)
    plt.text(13.5, 40, r'\textbf{L1}')
    plt.plot([left_top[0], right_top[0]], [left_top[1], right_top[1]], color = 'black', linewidth = 6)
    plt.text(13.5, 47, r'\textbf{L2}')
    plt.plot([left_bottom[0], left_top[0]], [0, left_top[1]], color = 'black', linewidth = 2, linestyle = '--')
    plt.plot([right_bottom[0], right_top[0]], [0, right_top[1]], color = 'black', linewidth = 2, linestyle = '--')
    plt.text(15.5, 15, r'\textbf{t=15}')
    plt.text(25.5, 15, r'\textbf{t=25}')
    plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
    plt.xlim((10, 30))
    plt.ylim((10, 70))
    plt.xlabel(r'\textbf{t (s)}')
    plt.ylabel(r'\textbf{X (m)}')
    plt.grid(True)
    plt.show(block = False)
    # plt.title(r'\textbf{X component of naval signals}')
    # naval_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval_x.eps', format='eps')
    # naval_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval_x.png', format='png')


    naval_y_fig = plt.figure()
    pos_counter, neg_counter = 0, 0
    for i in range(num_signals):
        if labels[i] > 0 and pos_counter <= 400:
            pos_counter += 1
            plt.plot(timepoints[:31], signals[i][1][:31], color='green', linewidth = 0.75, label=r'\textbf{Positive Signals}')
        elif labels[i] < 0 and neg_counter <= 400:
            neg_counter +=1
            plt.plot(timepoints[:31], signals[i][1][:31], color='red', linewidth = 0.75, label=r'\textbf{Negative Signals}')
    # plt.legend(loc = 'upper right')
    left_bottom, right_bottom = [12, 26], [20, 26]
    left_top, right_top = [12, 32], [20, 32]
    plt.plot([left_bottom[0], right_bottom[0]], [left_bottom[1], right_bottom[1]], color = 'black', linewidth = 5)
    plt.text(10, 26, r'\textbf{L3}')
    plt.plot([left_top[0], right_top[0]], [left_top[1], right_top[1]], color = 'black', linewidth = 5)
    plt.text(10, 31, r'\textbf{L4}')
    plt.plot([left_bottom[0], left_top[0]], [0, left_top[1]], color = 'black', linewidth = 2, linestyle = '--')
    plt.plot([right_bottom[0], right_top[0]], [0, right_top[1]], color = 'black', linewidth = 2, linestyle = '--')
    plt.text(12.5, 15, r'\textbf{t=12}')
    plt.text(20.5, 15, r'\textbf{t=20}')
    plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
    plt.xlim((5, 25))
    plt.ylim((10, 45))
    plt.xlabel(r'\textbf{t (s)}')
    plt.ylabel(r'\textbf{Y (m)}')
    plt.grid(True)
    plt.show(block = False)
    # plt.title(r'\textbf{Y component of naval signals}')
    # naval_y_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval_y.eps', format='eps')
    # naval_y_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/naval_y.png', format='png')



    # rotated_signals = rotate_signals(signals)
    # data_dict = {'data':rotated_signals, 't':timepoints, 'labels':labels}
    # savemat("rotated_naval.mat",data_dict)




    # rotatedfig = plt.figure()
    # for i in range(num_signals):
    #     if labels[i] > 0:
    #         plt.plot(rotated_signals[i][0], rotated_signals[i][1], color = 'green', linewidth=1, label=r'\textbf{Positive Signals}')
    #     else:
    #         plt.plot(rotated_signals[i][0], rotated_signals[i][1], color = 'red', linewidth=1, label=r'\textbf{Negative Signals}')
    # plt.legend(loc = 'lower right')
    # plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
    # plt.xlabel(r'\textbf{X (m)}')
    # plt.ylabel(r'\textbf{Y (m)}')
    # plt.grid(True)
    # plt.show(block = False)
    # plt.title(r'\textbf{Rotated Naval Scenario}')
    # rotatedfig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/rotated.eps', format='eps')
    # rotatedfig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/rotated.png', format='png')
    #
    #
    #
    #
    # rotated_x_fig = plt.figure()
    # for i in range(num_signals):
    #     if labels[i] > 0:
    #         plt.plot(timepoints, rotated_signals[i][0], color = 'green', linewidth=1, label=r'\textbf{Positive Signals}')
    #     else:
    #         plt.plot(timepoints, rotated_signals[i][0], color = 'red', linewidth=1, label=r'\textbf{Negative Signals}')
    # plt.legend(loc = 'lower right')
    # plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
    # plt.xlabel(r'\textbf{t (s)}')
    # plt.ylabel(r'\textbf{X (m)}')
    # plt.grid(True)
    # plt.show(block = False)
    # plt.title(r'\textbf{X component of rotated naval signals}')
    # rotated_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/rotated_x.eps', format='eps')
    # rotated_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/rotated_x.png', format='png')



    # rotated_y_fig = plt.figure()
    # for i in range(num_signals):
    #     if labels[i] > 0:
    #         plt.plot(timepoints, rotated_signals[i][1], color = 'green', linewidth=1, label=r'\textbf{Positive Signals}')
    #     else:
    #         plt.plot(timepoints, rotated_signals[i][1], color = 'red', linewidth=1, label=r'\textbf{Negative Signals}')
    # plt.legend(loc = 'lower right')
    # plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
    # plt.xlabel(r'\textbf{t (s)}')
    # plt.ylabel(r'\textbf{Y (m)}')
    # plt.grid(True)
    # plt.show(block = False)
    # plt.title(r'\textbf{Y component of rotated naval signals}')
    # rotated_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/rotated_x.eps', format='eps')
    # rotated_x_fig.savefig('/home/erfan/Documents/University/Projects/Learning_Specifications/nonincremental_tli/figures/rotated_x.png', format='png')




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
