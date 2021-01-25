import pickle
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import savemat
import random

plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']


plt.rcParams['text.usetex'] = True
plt.rcParams['font.weight'] = 'black'
plt.rcParams['font.size'] = '18'

##########################################################################
#### FINDING THE MAXIMUM AND MINIMUM TIME LENGTH OF SIGNALS
# time_lengths = []
# for i in range(1,151):
#     filename = "data_set/neg_label{}.pickle".format(i)
#     pickle_in = open(filename, "rb")
#     example_dict = pickle.load(pickle_in)
#     time = example_dict['time_history']
#
#     time_lengths.append(len(time))
#
# for i in range(1,151):
#     filename = "data_set/pos_label{}.pickle".format(i)
#     pickle_in = open(filename, "rb")
#     example_dict = pickle.load(pickle_in)
#     time = example_dict['time_history']
#
#     time_lengths.append(len(time))
#
# print("min length:", min(time_lengths))
# print("max length:", max(time_lengths))
##########################################################################

data = []
labels = []
t = []

### SETTING UP "t"
for i in range(500):
    t.append(i)


### SETTING UP "labels"
for i in range(150):
    labels.append(-1)

for i in range(150):
    labels.append(+1)

### SETTING UP "data"
for i in range(1,151):
    filename = "data_set/neg_label{}.pickle".format(i)
    pickle_in = open(filename, "rb")
    example_dict = pickle.load(pickle_in)

    player_x = np.array(example_dict['player_x'])
    player_y = np.array(example_dict['player_y'])
    player_z = np.array(example_dict['player_z'])
    player_v_x = np.array(example_dict['player_v_x'])
    player_v_y = np.array(example_dict['player_v_y'])
    player_v_z = np.array(example_dict['player_v_z'])

    other_x = np.array(example_dict['other_x'])
    other_y = np.array(example_dict['other_y'])
    other_z = np.array(example_dict['other_z'])
    other_v_x = np.array(example_dict['other_v_x'])
    other_v_y = np.array(example_dict['other_v_y'])
    other_v_z = np.array(example_dict['other_v_z'])

    x_rel = abs(player_x - other_x)
    y_rel = abs(player_y - other_y)
    z_rel = abs(player_z - other_z)

    v_x_rel = abs(player_v_x - other_v_x)
    v_y_rel = abs(player_v_y - other_v_y)
    v_z_rel = abs(player_v_z - other_v_z)

    x_rel_resample = signal.resample(x_rel, 520)
    x_rel = x_rel_resample[10:510]
    y_rel_resample = signal.resample(y_rel, 520)
    y_rel = y_rel_resample[10:510]
    z_rel_resample = signal.resample(z_rel, 520)
    z_rel = z_rel_resample[10:510]

    v_x_rel_resample = signal.resample(v_x_rel, 520)
    v_x_rel = v_x_rel_resample[10:510]
    v_y_rel_resample = signal.resample(v_y_rel, 520)
    v_y_rel = v_y_rel_resample[10:510]
    v_z_rel_resample = signal.resample(v_z_rel, 520)
    v_z_rel = v_z_rel_resample[10:510]

    data_sample = []
    data_sample.append(y_rel)
    data_sample.append(z_rel)
    data_sample.append(v_y_rel)
    data_sample.append(v_z_rel)
    data.append(data_sample)


for i in range(1,151):
    filename = "data_set/pos_label{}.pickle".format(i)
    pickle_in = open(filename, "rb")
    example_dict = pickle.load(pickle_in)

    player_x = np.array(example_dict['player_x'])
    player_y = np.array(example_dict['player_y'])
    player_z = np.array(example_dict['player_z'])
    player_v_x = np.array(example_dict['player_v_x'])
    player_v_y = np.array(example_dict['player_v_y'])
    player_v_z = np.array(example_dict['player_v_z'])

    other_x = np.array(example_dict['other_x'])
    other_y = np.array(example_dict['other_y'])
    other_z = np.array(example_dict['other_z'])
    other_v_x = np.array(example_dict['other_v_x'])
    other_v_y = np.array(example_dict['other_v_y'])
    other_v_z = np.array(example_dict['other_v_z'])

    x_rel = abs(player_x - other_x)
    y_rel = abs(player_y - other_y)
    z_rel = abs(player_z - other_z)

    v_x_rel = abs(player_v_x - other_v_x)
    v_y_rel = abs(player_v_y - other_v_y)
    v_z_rel = abs(player_v_z - other_v_z)

    x_rel_resample = signal.resample(x_rel, 520)
    x_rel = x_rel_resample[10:510]
    y_rel_resample = signal.resample(y_rel, 520)
    y_rel = y_rel_resample[10:510]
    z_rel_resample = signal.resample(z_rel, 520)
    z_rel = z_rel_resample[10:510]

    v_x_rel_resample = signal.resample(v_x_rel, 520)
    v_x_rel = v_x_rel_resample[10:510]
    v_y_rel_resample = signal.resample(v_y_rel, 520)
    v_y_rel = v_y_rel_resample[10:510]
    v_z_rel_resample = signal.resample(v_z_rel, 520)
    v_z_rel = v_z_rel_resample[10:510]

    data_sample = []
    data_sample.append(y_rel)
    data_sample.append(z_rel)
    data_sample.append(v_y_rel)
    data_sample.append(v_z_rel)
    data.append(data_sample)

temp = list(zip(data, labels))
random.shuffle(temp)
res1, res2 = zip(*temp)
data = list(res1)
labels = list(res2)
data_dict = {"data":data, "labels":labels, "t":t}
savemat("carla_scenario.mat", data_dict)


##########################################################################
###### PLOTTING SOME RESULTS

# filename = "data_set/pos_label10.pickle"
# pickle_in = open(filename, "rb")
# example_dict = pickle.load(pickle_in)
#
# time = example_dict['time_history']
#
# player_x = np.array(example_dict['player_x'])
# player_y = np.array(example_dict['player_y'])
# player_z = np.array(example_dict['player_z'])
# player_v_x = np.array(example_dict['player_v_x'])
# player_v_y = np.array(example_dict['player_v_y'])
# player_v_z = np.array(example_dict['player_v_z'])
#
# other_x = np.array(example_dict['other_x'])
# other_y = np.array(example_dict['other_y'])
# other_z = np.array(example_dict['other_z'])
# other_v_x = np.array(example_dict['other_v_x'])
# other_v_y = np.array(example_dict['other_v_y'])
# other_v_z = np.array(example_dict['other_v_z'])
#
# x_rel = abs(player_x - other_x)
# y_rel = abs(player_y - other_y)
# z_rel = abs(player_z - other_z)
#
# v_x_rel = abs(player_v_x - other_v_x)
# v_y_rel = abs(player_v_y - other_v_y)
# v_z_rel = abs(player_v_z - other_v_z)
#
#
#
# f = signal.resample(x_rel, 500)
# t_new = np.linspace(0, time[-1], 500)
# relative_x_fig  = plt.figure()
# plt.plot(time, x_rel, 'go-', t_new, f, '.-')
# # plt.plot(time, x_rel, color = 'red', linewidth = 4, label = r'\textbf{relative x}')
# plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
# plt.legend(['data', 'resampled'], loc = 'best')
# plt.xlabel(r'\textbf{Time (s)}')
# plt.ylabel(r'\textbf{Relative x (m)}')
# plt.grid(True)
#
#
# f = signal.resample(y_rel, 500)
# t_new = np.linspace(0, time[-1], 500)
# relative_y_fig  = plt.figure()
# plt.plot(time, y_rel, 'go-', t_new, f, '.-')
# # plt.plot(time, y_rel, color = 'red', linewidth = 4, label = r'\textbf{relative y}')
# plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
# plt.legend(['data', 'resampled'], loc = 'best')
# plt.xlabel(r'\textbf{Time (s)}')
# plt.ylabel(r'\textbf{Relative y (m)}')
# plt.grid(True)
#
#
# f = signal.resample(z_rel, 500)
# t_new = np.linspace(0, time[-1], 500)
# relative_z_fig  = plt.figure()
# plt.plot(time, z_rel, 'go-', t_new, f, '.-')
# # plt.plot(time, z_rel, color = 'red', linewidth = 4, label = r'\textbf{relative z}')
# plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
# plt.legend(['data', 'resampled'], loc = 'best')
# plt.xlabel(r'\textbf{Time (s)}')
# plt.ylabel(r'\textbf{Relative z (m)}')
# plt.grid(True)
#
#
# f = signal.resample(v_x_rel, 500)
# t_new = np.linspace(0, time[-1], 500)
# relative_v_x_fig  = plt.figure()
# plt.plot(time, v_x_rel, 'go-', t_new, f, '.-')
# # plt.plot(time, v_x_rel, color = 'red', linewidth = 4, label = r'\textbf{relative v_x}')
# plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
# plt.legend(['data', 'resampled'], loc = 'best')
# plt.xlabel(r'\textbf{Time (s)}')
# plt.ylabel(r'\textbf{Relative v_x (m/s)}')
# plt.grid(True)
#
#
# f = signal.resample(v_y_rel, 500)
# t_new = np.linspace(0, time[-1], 500)
# relative_v_y_fig  = plt.figure()
# plt.plot(time, v_y_rel, 'go-', t_new, f, '.-')
# # plt.plot(time, v_y_rel, color = 'red', linewidth = 4, label = r'\textbf{relative v_y}')
# plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
# plt.legend(['data', 'resampled'], loc = 'best')
# plt.xlabel(r'\textbf{Time (s)}')
# plt.ylabel(r'\textbf{Relative v_y (m/s)}')
# plt.grid(True)
#
#
# f = signal.resample(v_z_rel, 500)
# t_new = np.linspace(0, time[-1], 500)
# relative_v_z_fig  = plt.figure()
# plt.plot(time, v_z_rel, 'go-', t_new, f, '.-')
# # plt.plot(time, v_z_rel, color = 'red', linewidth = 4, label = r'\textbf{relative v_z}')
# plt.tick_params(axis='both', labelsize = 22, labelcolor = 'black')
# plt.legend(['data', 'resampled'], loc = 'best')
# plt.xlabel(r'\textbf{Time (s)}')
# plt.ylabel(r'\textbf{Relative v_z (m/s)}')
# plt.grid(True)
#
#
# plt.show()
