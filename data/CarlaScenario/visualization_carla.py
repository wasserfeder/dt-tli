from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
plt.rcParams['text.usetex'] = True
plt.rcParams['font.weight'] = 'black'
plt.rcParams['font.size'] = '18'


filename = 'carla_scenario_300.mat'

mat_data        = loadmat(filename)
timepoints      = mat_data['t'][0]
labels          = mat_data['labels'][0]
signals         = mat_data['data']
num_signals     = len(signals)
pos_indices, neg_indices = [], []
for i in range(len(labels)):
    if labels[i] >= 0:
        pos_indices.append(i)
    else:
        neg_indices.append(i)


positivefig, axs = plt.subplots(2, 2)
for i in pos_indices:
    for j in range(2):
        axs[j, 0].plot(timepoints, signals[i][j])
        axs[j, 0].grid(True)
    for k in range(2):
        axs[k, 1].plot(timepoints, signals[i][k + 2])
        axs[k, 1].grid(True)
    axs[0, 0].set_ylabel('Relative y (m)')
    axs[1, 0].set_ylabel('Relative z (m)')
    axs[0, 1].set_ylabel('Relative v-y (m/s)')
    axs[1, 1].set_ylabel('Relative v-z (m/s)')
    axs[1,0].set_xlabel('Time')
    axs[1,1].set_xlabel('Time')
positivefig.suptitle('Positive labels', fontsize = 16)


negativefig, axs = plt.subplots(2, 2)
for i in neg_indices:
    for j in range(2):
        axs[j, 0].plot(timepoints, signals[i][j])
        axs[j, 0].grid(True)
    for k in range(2):
        axs[k, 1].plot(timepoints, signals[i][k + 2])
        axs[k, 1].grid(True)
    axs[0, 0].set_ylabel('Relative y (m)')
    axs[1, 0].set_ylabel('Relative z (m)')
    axs[0, 1].set_ylabel('Relative v-y (m/s)')
    axs[1, 1].set_ylabel('Relative v-z (m/s)')
    axs[1,0].set_xlabel('Time')
    axs[1,1].set_xlabel('Time')
negativefig.suptitle('Negative labels', fontsize = 16)

plt.show()
