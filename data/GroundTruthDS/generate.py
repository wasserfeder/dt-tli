'''
Created on Mar 5, 2018

@author: cristi
'''

from datetime import date

import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml


def refine_trace(p1, p2, p3, p4, nsamples):
    '''Performs interpolation between the four points such that the first 10%
    of points are in the first second, 60% in the next 8.5 seconds, and the
    rest in the final second.

    Note: Assumes that `nsamples` is a multiple of 10.
    '''
    points = np.array([p1, p2, p3, p4])
    tck, _ = sp.interpolate.splprep((points[:, 0], points[:, 1]), s=0)
    param = []
    param.extend(np.linspace(0.00, 0.05, np.floor(0.1*nsamples),
                             endpoint=False))
    param.extend(np.linspace(0.05, 0.9, np.floor(0.6*nsamples),
                             endpoint=False))
    param.extend(np.linspace(0.9, 1.00, np.ceil(0.3*nsamples)+1))
    assert len(param) == nsamples+1

    return sp.interpolate.splev(param, tck)

def generate_traces(n_pos_traces=50, n_neg_traces=None, nsamples=20, seed=1001,
                    datafile='data', figname='data.png'):
    '''TODO: add doc string
    '''
    # to make things simple we assumet that nsamples is a multiple of 10
    assert 10*int(0.1*nsamples) == nsamples, 'nsamples is not a multiple of 10!'

    if n_neg_traces is None:
        n_neg_traces = 2*n_pos_traces

    # formula used for this example
    phi = ('G[0, 1]((x < 0.1) && (y < 0.6) && (y >= 0.4))'
           '&& G[7, 10]((x >= 0.7) && ((y >= 0.8) || (y < 0.2)))')
    # the time bounds of the formula as indices
    sec0 = 0
    sec1 = int(0.1*nsamples)
    sec7 = int(0.7*nsamples)
    sec10 = nsamples
    time_bounds = [sec0, sec1, sec7, sec10]

    # set seed for reproducible data set
    npr.seed(seed)

    # generate positive traces
    positive = []
    dev = 0.02
    for _ in range(n_pos_traces):
        p1 = np.array([npr.uniform(0, 0.1), npr.uniform(0.47, 0.53)])
        p2 = np.array([npr.uniform(0.3, 0.5), npr.uniform(0.4, 0.6)])
        if npr.random() < 0.5: # upper end region
            ly, hy = 0.8+dev, 1.0-dev
        else: # lower end region
            ly, hy = 0.0+dev, 0.2-dev
        p3 = np.array([0.7, npr.uniform(ly, hy)])
        p4 = np.array([npr.uniform(0.8, 1), npr.uniform(ly, hy)])

        trace = refine_trace(p1, p2, p3, p4, nsamples)
        # sanity checks to ensure that the signal is in the start and end
        # regions at the corresponding time intervals
        assert [0.0 <= x < 0.1 for x in trace[0][:sec1]]
        assert [0.4 <= y < 0.6 for y in trace[1][:sec1]]
        assert [0.7 <= x < 1.0 for x in trace[0][sec7:]]
        assert [0.0 <= y < 0.2 or 0.8 <= y < 1.0 for y in trace[1][sec7:]]
        positive.append(trace)

    # generate negative traces
    negative1 = []
    negative2 = []
    for _ in range(n_neg_traces):
        if npr.random() < 0.5: # not in start region
            if npr.random() < 0.5: # below start region
                ly, hy = 0.0+dev, 0.4-dev
            else: # above start region
                ly, hy = 0.6+dev, 1.0-dev
            p1 = np.array([npr.uniform(0, 0.1), npr.uniform(ly, hy)])
            p2 = np.array([npr.uniform(0.3, 0.5), npr.uniform(ly, hy)])
            if npr.random() < 0.5: # end anywhere
                p3 = np.array([0.7, npr.uniform(0.1, 0.9)])
                p4 = np.array([npr.uniform(0.8, 1.0), npr.uniform(0.1, 0.9)])
            else: # end in a end regions
                if npr.random() < 0.5: # upper end region
                    ly, hy = 0.8+dev, 1.0-dev
                else: # lower end region
                    ly, hy = 0.0+dev, 0.2-dev
                p3 = np.array([0.7, npr.uniform(ly, hy)])
                p4 = np.array([npr.uniform(0.8, 1), npr.uniform(ly, hy)])

            trace = refine_trace(p1, p2, p3, p4, nsamples)
            # sanity checks to ensure that the signal is not in the start region
            # at the corresponding time interval
            assert [0.0 <= x < 0.1 for x in trace[0][:sec1]]
            assert [y < 0.4 or y>= 0.6 for y in trace[1][:sec1]]
            negative1.append(trace)
        else: # not in end regions
            if npr.random() < 0.5: # start anywhere
                p1 = np.array([npr.uniform(0, 0.1), npr.uniform(0.1, 0.9)])
                p2 = np.array([npr.uniform(0.3, 0.5), npr.uniform(0.1, 0.9)])
            else: # in start region
                p1 = np.array([npr.uniform(0, 0.1), npr.uniform(0.47, 0.53)])
                p2 = np.array([npr.uniform(0.3, 0.5), npr.uniform(0.4, 0.6)])
            p3 = np.array([0.7, npr.uniform(0.2, 0.8)])
            p4 = np.array([npr.uniform(0.8, 1.0), npr.uniform(0.2, 0.8)])

            trace = refine_trace(p1, p2, p3, p4, nsamples)
            # sanity checks to ensure that the signal is not in end regions
            # at the corresponding time interval
            assert [0.7 <= x < 1.0 for x in trace[0][sec7:]]
            assert [0.2 <= y < 0.8 for y in trace[1][sec7:]]
            negative2.append(trace)

    # sanity check to make sure all signals are in the unit cube for all times
    for trace in positive + negative1 + negative2:
        assert [0.0 <= x < 1.0 for x in trace[0]]
        assert [0.0 <= y < 1.0 for y in trace[1]]

    # save traces to yaml file
    denumpify = lambda dataset: [{'x': map(float, trace[0]),
                                'y': map(float, trace[1])} for trace in dataset]
    with open(datafile + '.yaml', 'w') as fout:
        yaml.dump({'positive' : denumpify(positive),
                   'negative' : denumpify(negative1 + negative2)},
                  fout)
    # save traces to mat file
    data = np.zeros((n_pos_traces + n_neg_traces, 2, nsamples+1))
    for k, trace in enumerate(positive + negative1 + negative2):
        data[k, 0] = trace[0]
        data[k, 1] = trace[1]
    labels = [1]*n_pos_traces + [-1]*n_neg_traces
    mat_data = {
        'data' : data,
        'labels' : labels,
        't': np.arange(nsamples+1)/2.0 #FIXME: assumes nsamples=20
    }
    scio.savemat(datafile + '.mat', mat_data)
    # save traces to csv format
    with open(datafile + '.csv', 'w') as fout:
        for k, trace in enumerate(positive):
            np.savetxt(fout, trace, delimiter=',',
                       header='positive trace: ' + str(k+1))
        for k, trace in enumerate(negative1 + negative2):
            np.savetxt(fout, trace, delimiter=',',
                       header='negative trace: ' + str(k+1))

    # plot signals and regions
    # plt.figure(figsize=(11, 6))
    plt.figure(figsize=(6, 6))
    plt.axis('equal')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    # plot start and end regions
    plt.fill([0, 0, 0.2, 0.2], [0.4, 0.6, 0.6, 0.4], color='dimgray')
    plt.fill([0.7, 0.7, 1, 1], [0.8, 1.0, 1.0, 0.8], color='black')
    plt.fill([0.7, 0.7, 1, 1], [0.0, 0.2, 0.2, 0.0], color='black')

    # plot signals
    datasets = ((positive, 'blue', 'Positive'),
                (negative1, 'red', 'Not in start region'),
                (negative2, 'orange', 'Not in end regions'))
    for dataset, color, _ in datasets:
        for trace in dataset:
            x, y = trace
            # plot trace
            plt.plot(x, y, '-', color=color)
            # plot locations associates with the time bounds
            plt.plot(x[time_bounds], y[time_bounds], 'D', color=color)

    # plt.legend([mpatches.Patch(color=c) for _, c, _ in datasets],
    #            [d for _, c, d in datasets], fontsize=26,
    #            bbox_to_anchor=(1.04, 1), borderaxespad=0)
    # plt.tight_layout(rect=[0,0,0.57,1])
    plt.tight_layout()
    # save figure
    plt.savefig(figname)
    plt.show()

if __name__ == '__main__':
    generate_traces()
