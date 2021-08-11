from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
plt.rcParams['text.usetex'] = True
plt.rcParams['font.weight'] = 'black'
plt.rcParams['font.size'] = '22'


filename = 'carla_scenario_300.mat'

mat_data        = loadmat(filename)
timepoints      = mat_data['t'][0]
labels          = mat_data['labels'][0]
signals         = mat_data['data']
num_signals     = len(signals)
pos_indices, neg_indices = [], []
pos_signals, neg_signals =[], []
for i in range(len(labels)):
    if labels[i] >= 0:
        pos_indices.append(i)
        pos_signals.append(signals[i])
    else:
        neg_indices.append(i)
        neg_signals.append(signals[i])

pos_signals = np.array(pos_signals)
neg_signals = np.array(neg_signals)


y_pos_max, z_pos_max = [], []
y_pos_mean, z_pos_mean = [], []
y_pos_min, z_pos_min = [], []
y_neg_max, z_neg_max = [], []
y_neg_mean, z_neg_mean = [], []
y_neg_min, z_neg_min = [], []

for t in timepoints:
    # y-variable
    y_pos_max.append(max(pos_signals[:, 0, t]))
    y_pos_mean.append(np.mean(pos_signals[:, 0, t]))
    y_pos_min.append(min(pos_signals[:, 0, t]))

    y_neg_max.append(max(neg_signals[:, 0, t]))
    y_neg_mean.append(np.mean(neg_signals[:, 0, t]))
    y_neg_min.append(min(neg_signals[:, 0, t]))

    # z-variable
    z_pos_max.append(max(pos_signals[:, 1, t]))
    z_pos_mean.append(np.mean(pos_signals[:, 1, t]))
    z_pos_min.append(min(pos_signals[:, 1, t]))

    z_neg_max.append(max(neg_signals[:, 1, t]))
    z_neg_mean.append(np.mean(neg_signals[:, 1, t]))
    z_neg_min.append(min(neg_signals[:, 1, t]))

## Position plot
y_fig, y_axs = plt.subplots(1, 2)
# y-variable
y_axs[0].plot(timepoints, y_pos_max, linewidth = 2, color = 'blue')
y_axs[0].plot(timepoints, y_pos_min, linewidth = 2, color = 'blue')
y_axs[0].plot(timepoints, y_pos_mean, linewidth = 2, color = 'blue', linestyle = '--')
y_axs[0].fill_between(timepoints, y_pos_max, y_pos_min, color = 'deepskyblue', alpha = 0.3)
y_axs[0].set_title(r'\textbf{With pedestrian}')
y_axs[0].set_ylabel(r'\textbf{y (m)}')
y_axs[0].set_xlabel(r'\textbf{Time (1/60 s)}')
y_axs[0].set_xlim((450, 500))
y_axs[0].set_ylim((-4, 65))
y_axs[0].hlines(y = 10.54, xmin = 450, xmax = 500, linestyle = '--', linewidth = 3)
y_axs[0].vlines(x = 473, ymin = -4, ymax = 65, linestyle = '--', linewidth = 3)
y_axs[0].vlines(x = 491, ymin = -4, ymax = 65, linestyle = '--', linewidth = 3)
y_axs[0].text(478, 13, r'\textbf{y = 10.54}', fontsize = 24, fontweight = 'bold')
y_axs[0].text(464, 2, r'\textbf{t = 473}', fontsize = 24, fontweight = 'bold')
y_axs[0].text(482, 2, r'\textbf{t = 491}', fontsize = 24, fontweight = 'bold')
y_axs[0].text(454, 40, r'\textbf{Satisfaction of}', fontsize = 24, fontweight = 'bold')
y_axs[0].text(451, 35, r'\textbf{F_{[473, 491]} (y < 10.54)}', fontsize = 24, fontweight = 'bold')
y_axs[0].annotate('', xy=(490, 11), xytext=(452, 38),
             arrowprops=dict(facecolor='black', shrink=0.25, width = 1),
             )
y_axs[0].grid(True)



y_axs[1].plot(timepoints, y_neg_max, linewidth = 2, color = 'red')
y_axs[1].plot(timepoints, y_neg_min, linewidth = 2, color = 'red')
y_axs[1].plot(timepoints, y_neg_mean, linewidth = 2, color = 'red', linestyle = '--')
y_axs[1].fill_between(timepoints, y_neg_max, y_neg_min, color = 'tomato', alpha = 0.3)
y_axs[1].set_title(r'\textbf{No pedestrian}')
y_axs[1].set_ylabel(r'\textbf{y (m)}')
y_axs[1].set_xlabel(r'\textbf{Time (1/60 s)}')
y_axs[1].set_xlim((450, 500))
y_axs[1].set_ylim((-4, 65))
y_axs[1].hlines(y = 10.54, xmin = 450, xmax = 500, linestyle = '--', linewidth = 3)
y_axs[1].vlines(x = 473, ymin = -4, ymax = 65, linestyle = '--', linewidth = 3)
y_axs[1].vlines(x = 491, ymin = -4, ymax = 65, linestyle = '--', linewidth = 3)
y_axs[1].text(478, 13, r'\textbf{y = 10.54}', fontsize = 24, fontweight = 'bold')
y_axs[1].text(464, 5, r'\textbf{t = 473}', fontsize = 24, fontweight = 'bold')
y_axs[1].text(482, 5, r'\textbf{t = 491}', fontsize = 24, fontweight = 'bold')
y_axs[1].text(454, 40, r'\textbf{Violation of}', fontsize = 24, fontweight = 'bold')
y_axs[1].text(451, 35, r'\textbf{F_{[473, 491]} (y < 10.54)}', fontsize = 24, fontweight = 'bold')
y_axs[1].annotate('', xy=(490, 29), xytext=(452, 33),
             arrowprops=dict(facecolor='black', shrink=0.25, width = 1),
             )
y_axs[1].grid(True)



z_fig, z_axs = plt.subplots(1, 2)
# z-variable
z_axs[0].plot(timepoints, z_pos_max, linewidth = 2, color = 'blue')
z_axs[0].plot(timepoints, z_pos_min, linewidth = 2, color = 'blue')
z_axs[0].plot(timepoints, z_pos_mean, linewidth = 2, color = 'blue', linestyle = '--')
z_axs[0].fill_between(timepoints, z_pos_max, z_pos_min, color = 'deepskyblue', alpha = 0.3)
z_axs[0].set_title(r'\textbf{With pedestrian}')
z_axs[0].set_ylabel(r'\textbf{z (m)}')
z_axs[0].set_xlabel(r'\textbf{Time (1/60 s)}')
z_axs[0].set_ylim((-0.2, 3))
z_axs[0].set_xlim((450, 500))
z_axs[0].hlines(y = 1.69, xmin = 450, xmax = 500, linestyle = '--', linewidth = 3)
z_axs[0].vlines(x = 470, ymin = -0.2, ymax = 3, linestyle = '--', linewidth = 3)
z_axs[0].vlines(x = 491, ymin = -0.2, ymax = 3, linestyle = '--', linewidth = 3)
z_axs[0].text(477, 1.8, r'\textbf{z = 1.69}', fontsize = 24, fontweight = 'bold')
z_axs[0].text(461, 2.6, r'\textbf{t = 470}', fontsize = 24, fontweight = 'bold')
z_axs[0].text(482, 2.6, r'\textbf{t = 491}', fontsize = 24, fontweight = 'bold')
z_axs[0].text(453, 0.4, r'\textbf{Satisfaction of}', fontsize = 24, fontweight = 'bold')
z_axs[0].text(450, 0.2, r'\textbf{G_{[470, 491]} (z < 1.69)}', fontsize = 24, fontweight = 'bold')
z_axs[0].annotate('', xy=(490, 0.9), xytext=(452, 0.5),
             arrowprops=dict(facecolor='black', shrink=0.25, width = 1),
             )
z_axs[0].grid(True)



z_axs[1].plot(timepoints, z_neg_max, linewidth = 2, color = 'red')
z_axs[1].plot(timepoints, z_neg_min, linewidth = 2, color = 'red')
z_axs[1].plot(timepoints, z_neg_mean, linewidth = 2, color = 'red', linestyle = '--')
z_axs[1].fill_between(timepoints, z_neg_max, z_neg_min, color = 'tomato', alpha = 0.3)
z_axs[1].set_title(r'\textbf{No pedestrian}')
z_axs[1].set_xlabel(r'\textbf{Time (1/60 s)}')
z_axs[1].set_ylabel(r'\textbf{z (m)}')
z_axs[1].set_ylim((-0.2, 3))
z_axs[1].set_xlim((450, 500))
z_axs[1].hlines(y = 1.69, xmin = 450, xmax = 500, linestyle = '--', linewidth = 3)
z_axs[1].vlines(x = 470, ymin = -0.2, ymax = 3, linestyle = '--', linewidth = 3)
z_axs[1].vlines(x = 491, ymin = -0.2, ymax = 3, linestyle = '--', linewidth = 3)
z_axs[1].text(475, 1.5, r'\textbf{z = 1.69}', fontsize = 24, fontweight = 'bold')
z_axs[1].text(461, 0.2, r'\textbf{t = 470}', fontsize = 24, fontweight = 'bold')
z_axs[1].text(482, 0.2, r'\textbf{t = 491}', fontsize = 24, fontweight = 'bold')
z_axs[1].text(473, 2.7, r'\textbf{Violation of}', fontsize = 24, fontweight = 'bold')
z_axs[1].text(470, 2.5, r'\textbf{G_{[470, 491]} (z < 1.69)}', fontsize = 24, fontweight = 'bold')
z_axs[1].annotate('', xy=(479, 1.8), xytext=(481, 2.6),
             arrowprops=dict(facecolor='black', shrink=0.25, width = 1),
             )
z_axs[1].grid(True)




vy_pos_max, vy_neg_max = [], []
vy_pos_mean, vy_neg_mean = [], []
vy_pos_min, vy_neg_min = [], []

for t in timepoints:
    # vy-variable
    vy_pos_max.append(max(pos_signals[:, 2, t]))
    vy_pos_mean.append(np.mean(pos_signals[:, 2, t]))
    vy_pos_min.append(min(pos_signals[:, 2, t]))

    vy_neg_max.append(max(neg_signals[:, 2, t]))
    vy_neg_mean.append(np.mean(neg_signals[:, 2, t]))
    vy_neg_min.append(min(neg_signals[:, 2, t]))

## Velocity plot
vy_fig, vy_axs = plt.subplots(1, 2)

vy_axs[0].plot(timepoints, vy_pos_max, linewidth = 2, color = 'blue')
vy_axs[0].plot(timepoints, vy_pos_min, linewidth = 2, color = 'blue')
vy_axs[0].plot(timepoints, vy_pos_mean, linewidth = 2, color = 'blue', linestyle = '--')
vy_axs[0].fill_between(timepoints, vy_pos_max, vy_pos_min, color = 'deepskyblue', alpha = 0.3)
vy_axs[0].set_title(r'\textbf{With pedestrian}')
vy_axs[0].set_ylabel(r'\textbf{v_y (m/s)}')
vy_axs[0].set_xlabel(r'\textbf{Time (1/60 s)}')
vy_axs[0].set_xlim((450, 500))
vy_axs[0].set_ylim((-2, 32))
vy_axs[0].hlines(y = 3.31, xmin = 450, xmax = 500, linestyle = '--', linewidth = 3)
vy_axs[0].vlines(x = 473, ymin = -2, ymax = 32, linestyle = '--', linewidth = 3)
vy_axs[0].vlines(x = 490, ymin = -2, ymax = 32, linestyle = '--', linewidth = 3)
vy_axs[0].text(478, 1.8, r'\textbf{v_y = 3.31}', fontsize = 24, fontweight = 'bold')
vy_axs[0].text(474, 27, r'\textbf{t = 473}', fontsize = 24, fontweight = 'bold')
vy_axs[0].text(491, 27, r'\textbf{t = 490}', fontsize = 24, fontweight = 'bold')
vy_axs[0].text(454, 25, r'\textbf{Satisfaction of}', fontsize = 24, fontweight = 'bold')
vy_axs[0].text(451, 23, r'\textbf{G_{[473, 490]} (v_y >= 3.31)}', fontsize = 22, fontweight = 'bold')
vy_axs[0].annotate('', xy=(490, 18), xytext=(452, 23),
             arrowprops=dict(facecolor='black', shrink=0.25, width = 1),
             )
vy_axs[0].grid(True)


vy_axs[1].plot(timepoints, vy_neg_max, linewidth = 2, color = 'red')
vy_axs[1].plot(timepoints, vy_neg_min, linewidth = 2, color = 'red')
vy_axs[1].plot(timepoints, vy_neg_mean, linewidth = 2, color = 'red', linestyle = '--')
vy_axs[1].fill_between(timepoints, vy_neg_max, vy_neg_min, color = 'tomato', alpha = 0.3)
vy_axs[1].set_title(r'\textbf{No pedestrian}')
vy_axs[1].set_ylabel(r'\textbf{v_y (m/s)}')
vy_axs[1].set_xlabel(r'\textbf{Time (1/60 s)}')
vy_axs[1].set_xlim((450, 500))
vy_axs[1].set_ylim((-2, 32))
vy_axs[1].hlines(y = 3.31, xmin = 450, xmax = 500, linestyle = '--', linewidth = 3)
vy_axs[1].vlines(x = 473, ymin = -2, ymax = 32, linestyle = '--', linewidth = 3)
vy_axs[1].vlines(x = 490, ymin = -2, ymax = 32, linestyle = '--', linewidth = 3)
vy_axs[1].text(478, 1.8, r'\textbf{v_y = 3.31}', fontsize = 24, fontweight = 'bold')
vy_axs[1].text(474, 26, r'\textbf{t = 473}', fontsize = 24, fontweight = 'bold')
vy_axs[1].text(491, 26, r'\textbf{t = 490}', fontsize = 24, fontweight = 'bold')
vy_axs[1].text(455, 18, r'\textbf{Violation of}', fontsize = 24, fontweight = 'bold')
vy_axs[1].text(451, 16, r'\textbf{G_{[473, 490]} (v_y >= 3.31)}', fontsize = 22, fontweight = 'bold')
vy_axs[1].annotate('', xy=(485, 0), xytext=(457, 19),
             arrowprops=dict(facecolor='black', shrink=0.25, width = 1),
             )
vy_axs[1].grid(True)




plt.show()
