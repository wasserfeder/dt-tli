#!/usr/bin/env python

'''TODO:
'''

import gurobipy as grb
from gurobipy import GRB
import numpy as np
import math
from scipy.io import loadmat, savemat
import argparse
from os import path
import os



######### (G_[14.6, 57.3] x_1 < 9.33)

class PrimitiveMILP(object):
    '''TODO:
    '''

    def __init__(self, signals, t1, t2, model=None):
        '''TODO:
        '''
        if model is not None:
            self.model = model
        else:
            self.model = grb.Model()

        self.variables = {}
        self.signals = signals
        self.rho_max = 12   # TODO: set proper value
        self.M = 20         # TODO: set proper value
        self.horizon = len(signals[0]) - 1
        self.num_sig = len(signals)

        min_sig, max_sig = np.zeros(self.num_sig), np.zeros(self.num_sig)
        for i in range(self.num_sig):
            min_sig[i] = min(signals[i])
            max_sig[i] = max(signals[i])
        min_thresh, max_thresh = min(min_sig), max(max_sig)

        ## primitive's Parameters
        self.threshold = self.model.addVar(name='threshold', lb=min_thresh,
                                            ub=max_thresh, vtype=GRB.CONTINUOUS)
        self.t1 = t1
        self.t2 = t2
        self.ind = [0]*(self.horizon+1)
        for t in range(self.horizon+1):
            if (self.t1 <= t) and (t <= self.t2):
                self.ind[t] = 1


    def predicate_robustness(self, signal_index, custom_encoding=False):
        '''TODO:
        '''
        signal = self.signals[signal_index]
        rho = self.model.addVar(name='r_pred_{}'.format(signal_index),
                                lb=-self.M, ub=self.M, vtype=GRB.CONTINUOUS)

        r_vars = [self.model.addVar(lb=-self.M, ub=self.M, vtype=GRB.CONTINUOUS)
                    for t in range(self.horizon+1)]
        s_vars = [self.model.addVar(lb=-self.M, ub=self.M, vtype=GRB.CONTINUOUS)
                    for t in range(self.horizon+1)]
        m_vars = [self.model.addVar(lb=-self.M, ub=self.M, vtype=GRB.CONTINUOUS)
                    for t in range(self.horizon+1)]

        for t, (r_var, s_var, m_var) in enumerate(zip(r_vars, s_vars, m_vars)):
            self.model.addConstr(s_var == (self.threshold - signal[t]))
            self.model.addConstr(m_var == self.M * (1 - 2 * self.ind[t]))
            self.model.addConstr(r_var == grb.max_(s_var, m_var))

        self.model.addConstr(rho == grb.min_(r_vars))
        return rho


    def pos_neg_partition(self, rho):
        self.S_t = [self.model.addVar(vtype=GRB.BINARY) for i in range(self.num_sig)]
        self.S_f = [self.model.addVar(vtype=GRB.BINARY) for i in range(self.num_sig)]
        for i in range(self.num_sig):
            self.model.addGenConstrIndicator(self.S_t[i], True, 0 <= rho[i])
            self.model.addGenConstrIndicator(self.S_t[i], False, rho[i] <= 0.001)
            self.model.addGenConstrIndicator(self.S_f[i], True, rho[i] <= 0.001)
            self.model.addGenConstrIndicator(self.S_f[i], False, 0 <= rho[i])


    def impurity_optimization(self, rho, pos_indices, neg_indices):
        self.S_tp = [self.model.addVar(vtype=GRB.BINARY) for i in range(self.num_sig)]
        self.S_tn = [self.model.addVar(vtype=GRB.BINARY) for i in range(self.num_sig)]
        self.S_fp = [self.model.addVar(vtype=GRB.BINARY) for i in range(self.num_sig)]
        self.S_fn = [self.model.addVar(vtype=GRB.BINARY) for i in range(self.num_sig)]

        for i in neg_indices:
            self.model.addConstr(self.S_tp[i] == 0)
            self.model.addConstr(self.S_tn[i] == self.S_t[i])
            self.model.addConstr(self.S_fp[i] == 0)
            self.model.addConstr(self.S_fn[i] == self.S_f[i])
        for i in pos_indices:
            self.model.addConstr(self.S_tp[i] == self.S_t[i])
            self.model.addConstr(self.S_tn[i] == 0)
            self.model.addConstr(self.S_fp[i] == self.S_f[i])
            self.model.addConstr(self.S_fn[i] == 0)


        # for i in range(self.num_sig):
        #     self.model.addConstr(self.S_tp[i] == grb.and_(self.S_t[i], pos_ind[i]))
        #     self.model.addConstr(self.S_tn[i] == grb.and_(self.S_t[i], neg_ind[i]))
        #     self.model.addConstr(self.S_fp[i] == grb.and_(self.S_f[i], pos_ind[i]))
        #     self.model.addConstr(self.S_fn[i] == grb.and_(self.S_f[i], neg_ind[i]))

        self.S_tp_card = self.model.addVar(lb=0, ub=self.num_sig, vtype=GRB.INTEGER)
        self.S_tn_card = self.model.addVar(lb=0, ub=self.num_sig, vtype=GRB.INTEGER)
        self.S_fp_card = self.model.addVar(lb=0, ub=self.num_sig, vtype=GRB.INTEGER)
        self.S_fn_card = self.model.addVar(lb=0, ub=self.num_sig, vtype=GRB.INTEGER)

        self.model.addConstr(self.S_tp_card == grb.quicksum(self.S_tp))
        self.model.addConstr(self.S_tn_card == grb.quicksum(self.S_tn))
        self.model.addConstr(self.S_fp_card == grb.quicksum(self.S_fp))
        self.model.addConstr(self.S_fn_card == grb.quicksum(self.S_fn))

        self.MR_true = self.model.addVar(lb=0, ub=self.num_sig, vtype=GRB.INTEGER)
        self.MR_false = self.model.addVar(lb=0, ub=self.num_sig, vtype=GRB.INTEGER)

        self.model.addConstr(self.MR_true == grb.min_(self.S_tp_card, self.S_tn_card))
        self.model.addConstr(self.MR_false == grb.min_(self.S_fp_card, self.S_fn_card))

        ################# Objective function #################
        self.model.setObjective(self.MR_true + self.MR_false, GRB.MINIMIZE)
        self.model.update()


    def get_interval(self):
        '''TODO:
        '''
        if self.model.status == GRB.OPTIMAL:
            values = np.array([var.x for var in self.ind])
            values = np.argwhere(values > 0.5)
            return np.min(values), np.max(values)
        else:
            raise RuntimeError('The model needs to be solved first!')


def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file', help='.mat file containing the data')
    return parser



def get_path(f):
    return path.join(os.getcwd(), f)



def test1():
    '''TODO:
    '''
    args            = get_argparser().parse_args()
    filename        = get_path(args.file)
    mat_data        = loadmat(filename)
    mat_data        = loadmat(filename)
    timepoints      = list(mat_data['t'][0])
    labels          = list(mat_data['labels'][0])
    num_sig         = len(labels)
    signals         = [mat_data['data'][i][0] for i in range(num_sig)]
    pos_indices, neg_indices = [], []
    for i in range(len(labels)):
        if labels[i] >= 0:
            pos_indices.append(i)
        else:
            neg_indices.append(i)


    # horizon = len(signals[0])-1
    # t = [0]*len(signals[0])
    # print(len(t))
    solutions = []
    for t1 in range(len(timepoints)-1):
        for t2 in range(t1+1, len(timepoints)):
            milp = PrimitiveMILP(signals, t1, t2, None)
            rho = [milp.predicate_robustness(i) for i in range(num_sig)]
            milp.pos_neg_partition(rho)
            milp.impurity_optimization(rho, pos_indices, neg_indices)

            milp.model.update()
            milp.model.optimize()
            obj = milp.model.getObjective()
            threshold = milp.threshold.x
            solutions.append([obj, threshold, t1, t2])
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(solutions)
    objs = []
    for solution in solutions:
        objs.append(solution[0])
    min_index = np.argmin(objs)
    print(solutions[min_index])

def main():
    test1()


if __name__ == '__main__':
    main()
