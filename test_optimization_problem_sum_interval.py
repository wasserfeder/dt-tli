#!/usr/bin/env python

'''TODO:
'''

import os
import argparse
import math
import time

import gurobipy as grb
from gurobipy import GRB
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

from robustness import trace_robustness_order1_lkt_1d
from robustness import trace_robustness_order1_lkt_nd
from robustness import traces_robustness_order1_lkt


class PrimitiveMILP(object):
    '''TODO:
    '''

    def __init__(self, signals, labels, ranges, model=None):
        '''TODO:
        '''
        if model is not None:
            self.model = model
        else:
            self.model = grb.Model()

        self.variables = {}
        self.signals = signals
        self.labels = labels
        self.M = 100        # TODO: set proper value
        self.horizon = len(signals[0][0])
        self.num_signals = len(signals)

        min_thresh = np.min(signals)
        max_thresh = np.max(signals)

        self.lookuptable = traces_robustness_order1_lkt(self.signals)

        # primitive's parameters
        self.threshold = self.model.addVar(name='threshold', lb=min_thresh,
                                            ub=max_thresh, vtype=GRB.CONTINUOUS)

        self.intervals = [[None] * self.horizon for _ in range(self.horizon)]
        mutex_interval = 0
        for t0 in range(self.horizon):
            for dt in range(self.horizon - t0):
                vname = 'interval_{}_{}'.format(t0, t0+dt)
                self.intervals[dt][t0] = self.model.addVar(name=vname,
                                                           vtype=GRB.BINARY)
                mutex_interval += self.intervals[dt][t0]

        self.model.addConstr(mutex_interval == 1,
                             name='mutual_exclusive_time_intervals')

        self.model.update()

    def robustness(self, signal_index, signal_dimension=0, upper_bound=None):
        '''TODO:
        '''
        signal = self.signals[signal_index][signal_dimension]
        lkt = self.lookuptable[signal_index][signal_dimension]
        if upper_bound is None:
            upper_bound = self.M
        assert upper_bound >= 0

        weighted_sum = 0
        for t0 in range(self.horizon):
            for dt in range(self.horizon - t0):
                weighted_sum += self.intervals[dt][t0] * lkt[dt][t0]

        vname = 'r_no_path_{}_{}'.format(signal_index, signal_dimension)
        primitive = self.model.addVar(name=vname, lb=-self.M, ub=self.M,
                                      vtype=GRB.CONTINUOUS)
        self.model.addConstr(primitive == weighted_sum - self.threshold)

        vname = 'r_primitive_{}_{}'.format(signal_index, signal_dimension)
        rho = self.model.addVar(name=vname, lb=-self.M, ub=self.M,
                                vtype=GRB.CONTINUOUS)
        self.model.addConstr(rho == grb.min_(primitive, upper_bound))

        # vname = 'z_primitive_{}_{}'.format(signal_index, signal_dimension)
        # z = self.model.addVar(name=vname, vtype=GRB.BINARY)
        # self.model.addConstr(rho <= z * self.M)
        # self.model.addConstr(-rho <= (1 - z) * self.M)

        # return rho, z

        return rho

    def minimum_sum(self, primitive_variables, labels):
        '''TODO:
        '''
        tp_vars = []
        fp_vars = []
        tn_vars = []
        fn_vars = []
        for rho, label in zip(primitive_variables, labels):
            pos_rho = self.model.addVar(lb=-self.M, ub=self.M,
                                        vtype=GRB.CONTINUOUS)
            self.model.addConstr(pos_rho == grb.max_(rho, 0))
            neg_rho = self.model.addVar(lb=-self.M, ub=self.M,
                                        vtype=GRB.CONTINUOUS)
            self.model.addConstr(neg_rho == grb.min_(rho, 0))
            if label > 0:
                tp_vars.append(pos_rho)
                fn_vars.append(neg_rho)
            else:
                fp_vars.append(pos_rho)
                tn_vars.append(neg_rho)

        tp = sum(tp_vars)
        fp = sum(fp_vars)
        tn = sum(tn_vars)
        fn = sum(fn_vars)

        psum = self.model.addVar(name='positive_sum', vtype=GRB.CONTINUOUS)
        self.model.addConstr(psum == tp - fn)
        nsum = self.model.addVar(name='negative_sum', vtype=GRB.CONTINUOUS)
        self.model.addConstr(nsum == fp - tn)
        tsum = self.model.addVar(name='correctly_classified_sum',
                                 vtype=GRB.CONTINUOUS)
        self.model.addConstr(tsum == tp - tn)
        msum = self.model.addVar(name='misclassified_sum', vtype=GRB.CONTINUOUS)
        self.model.addConstr(msum == fp - fn)

        var = self.model.addVar(name='objective', vtype=GRB.CONTINUOUS)
        self.model.addConstr(var == grb.min_(psum, nsum, tsum, msum))

        return var

    def impurity_optimization(self, signal_dimension=0):
        '''TODO:
        '''
        primitive_variables = [self.robustness(i, signal_dimension)
                               for i in range(self.num_signals)]

        var = self.minimum_sum(primitive_variables, self.labels)

        # objective function
        self.model.setObjective(var, GRB.MINIMIZE)
        self.model.update()

    def get_interval(self):
        '''TODO:
        '''
        if self.model.status == GRB.OPTIMAL:
            check_solution = 0
            interval = None
            for t0 in range(self.horizon):
                for dt in range(self.horizon - t0):
                    check_solution += self.intervals[dt][t0].X
                    if self.intervals[dt][t0].X:
                        interval = (t0, t0+dt)
            assert check_solution == 1
            return interval
        else:
            raise RuntimeError('The model needs to be solved first!')

    def get_treshold(self):
        if self.model.status == GRB.OPTIMAL:
            return self.threshold.X
        else:
            raise RuntimeError('The model needs to be solved first!')


def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file', help='.mat file containing the data')
    return parser


def get_path(f):
    return os.path.join(os.getcwd(), f)


def test1():
    '''TODO:
    '''
    args            = get_argparser().parse_args()
    filename        = get_path(args.file)
    mat_data        = loadmat(filename)
    timepoints      = mat_data['t'][0]
    labels          = mat_data['labels'][0]
    signals         = mat_data['data']
    print(signals.shape)
    print('Number of signals:', len(signals))
    print('Time points:', len(timepoints))

    # signals = -signals

    t0 = time.time()
    milp = PrimitiveMILP(signals, labels, None)
    # milp.impurity_optimization(signal_dimension=0) # x-axis
    milp.impurity_optimization(signal_dimension=1) # y-axis

    dt = time.time() - t0
    print('Setup time:', dt)

    t0 = time.time()
    milp.model.optimize()
    dt = time.time() - t0
    print('Runtime:', dt)
    print('Model optimization status:', milp.model.status)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\tSolution:')
    print('Threshold:', milp.get_treshold())
    print('Time interval:', milp.get_interval())


def main():
    test1()

if __name__ == '__main__':
    main()