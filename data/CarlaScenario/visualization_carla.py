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


positionfig, axs = plt.subplots(2, 2)
for j in range(2):
    for i in pos_indices:
        axs[j, 0].plot(timepoints, signals[i][j])
        axs[j, 0].grid(True)


axs[0, 0].set_title("Positive Signals")
axs[0, 0].set_ylabel('Relative y (m)')
axs[1, 0].set_ylabel('Relative z (m)')
axs[1, 0].set_xlabel('Time')

for j in range(2):
    for i in neg_indices:
        axs[j, 1].plot(timepoints, signals[i][j])
        axs[j, 1].grid(True)

axs[0, 1].set_title("Negative Signals")
axs[1, 1].set_xlabel('Time')


# plt.show(block = True)
positionfig.savefig('/home/erfan/Documents/University/Projects/Learning Specifications/dt-tli/Figures/carla_pose_data.eps', format='eps')
positionfig.savefig('/home/erfan/Documents/University/Projects/Learning Specifications/dt-tli/Figures/carla_pose_data.png', format='png')


velfig, axs = plt.subplots(2, 2)
for j in range(2):
    for i in pos_indices:
        axs[j, 0].plot(timepoints, signals[i][j + 2])
        axs[j, 0].grid(True)


axs[0, 0].set_title("Positive Signals")
axs[0, 0].set_ylabel('Relative v-y (m/s)')
axs[1, 0].set_ylabel('Relative v-z (m/s)')
axs[1, 0].set_xlabel('Time')

for j in range(2):
    for i in neg_indices:
        axs[j, 1].plot(timepoints, signals[i][j + 2])
        axs[j, 1].grid(True)

axs[0, 1].set_title("Negative Signals")
axs[1, 1].set_xlabel('Time')


# plt.show(block = True)
velfig.savefig('/home/erfan/Documents/University/Projects/Learning Specifications/dt-tli/Figures/carla_vel_data.eps', format='eps')
velfig.savefig('/home/erfan/Documents/University/Projects/Learning Specifications/dt-tli/Figures/carla_vel_data.png', format='png')



# positivefig, axs = plt.subplots(2, 2)
# for i in pos_indices:
#     for j in range(2):
#         axs[j, 0].plot(timepoints, signals[i][j])
#         axs[j, 0].grid(True)
#     for k in range(2):
#         axs[k, 1].plot(timepoints, signals[i][k + 2])
#         axs[k, 1].grid(True)
#     axs[0, 0].set_ylabel('Relative y (m)')
#     axs[1, 0].set_ylabel('Relative z (m)')
#     axs[0, 1].set_ylabel('Relative v-y (m/s)')
#     axs[1, 1].set_ylabel('Relative v-z (m/s)')
#     axs[1,0].set_xlabel('Time')
#     axs[1,1].set_xlabel('Time')
# positivefig.suptitle('Positive labels', fontsize = 16)
# positivefig.savefig('/home/erfan/Documents/University/Projects/Learning Specifications/dt-tli/Figures/carla_pos_data.eps', format='eps')
# positivefig.savefig('/home/erfan/Documents/University/Projects/Learning Specifications/dt-tli/Figures/carla_pos_data.png', format='png')
#
#
# negativefig, axs = plt.subplots(2, 2)
# for i in neg_indices:
#     for j in range(2):
#         axs[j, 0].plot(timepoints, signals[i][j])
#         axs[j, 0].grid(True)
#     for k in range(2):
#         axs[k, 1].plot(timepoints, signals[i][k + 2])
#         axs[k, 1].grid(True)
#     axs[0, 0].set_ylabel('Relative y (m)')
#     axs[1, 0].set_ylabel('Relative z (m)')
#     axs[0, 1].set_ylabel('Relative v-y (m/s)')
#     axs[1, 1].set_ylabel('Relative v-z (m/s)')
#     axs[1,0].set_xlabel('Time')
#     axs[1,1].set_xlabel('Time')
# negativefig.suptitle('Negative labels', fontsize = 16)
# negativefig.savefig('/home/erfan/Documents/University/Projects/Learning Specifications/dt-tli/Figures/carla_neg_data.eps', format='eps')
# negativefig.savefig('/home/erfan/Documents/University/Projects/Learning Specifications/dt-tli/Figures/carla_neg_data.png', format='png')

plt.show()
