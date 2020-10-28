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

    def __init__(self, signals, ranges, model=None):
        '''TODO:
        '''
        if model is not None:
            self.model = model
        else:
            self.model = grb.Model()

        self.variables = {}

        self.signals = signals

        self.rho_max = 12 # TODO: set proper value
        self.M = 20 # TODO: set proper value
        self.horizon = len(self.signals[0]) - 1

        self.num_signals = len(signals)
        min_signals, max_signals = np.zeros(self.num_signals), np.zeros(self.num_signals)
        for i in range(self.num_signals):
            min_signals[i] = min(signals[i])
            max_signals[i] = max(signals[i])
        min_threshold, max_threshold = min(min_signals), max(max_signals)
        print(min_threshold, max_threshold)
        # self.threshold = 11.16

        ## primitive threshold
        self.threshold = self.model.addVar(name='threshold',
                                           lb=min_threshold, ub=max_threshold,
                                           vtype=GRB.CONTINUOUS)
        # primitive time interval indicator function
        self.indicator = np.zeros(self.horizon + 1)
        for t in range(15, 57):
            self.indicator[t] = 1
        # self.indicator = self.indicator_function()



    def indicator_function(self):
        '''TODO:
        '''
        ind = [self.model.addVar(name='ind_{}'.format(t), vtype=GRB.BINARY)
               for t in range(self.horizon+1)]
        ind_inc = [self.model.addVar(name='ind_inc_{}'.format(t),
                                     vtype=GRB.BINARY)
                   for t in range(self.horizon+1)]
        ind_dec = [self.model.addVar(name='ind_dec_{}'.format(t),
                                     vtype=GRB.BINARY)
                   for t in range(self.horizon+1)]

        # interval constraints
        self.model.addConstrs(ind[t] == grb.min_(ind_inc[t], ind_dec[t])
                              for t in range(self.horizon+1))
        # monotonicity (detection) constraints
        self.model.addConstrs(ind_inc[t] <= ind_inc[t+1]
                              for t in range(self.horizon))
        self.model.addConstrs(ind_dec[t] >= ind_dec[t+1]
                              for t in range(self.horizon))
        # non-degenerate interval constraint
        self.model.addConstr(sum(ind) >= 1)

        return ind

    def get_interval(self):
        '''TODO:
        '''
        if self.model.status == GRB.OPTIMAL:
            values = np.array([var.x for var in self.indicator])
            values = np.argwhere(values > 0.5)
            return np.min(values), np.max(values)
        else:
            raise RuntimeError('The model needs to be solved first!')

    def predicate_robustness(self, signal_index, custom_encoding=False):
        '''TODO:
        '''
        signal = self.signals[signal_index]
        rho = self.model.addVar(name='r_pred_{}'.format(signal_index),
                                lb=-self.M, ub=self.M,
                                vtype=GRB.CONTINUOUS)
        r_vars = [self.model.addVar(lb=-self.M, ub=self.M, vtype=GRB.CONTINUOUS)
             for t in range(self.horizon+1)]

        
        s_vars = [self.model.addVar(lb=-self.rho_max, ub=self.rho_max, vtype=GRB.CONTINUOUS)
                    for t in range(self.horizon+1)]
        m_vars = [self.model.addVar(lb=-self.M, ub=self.M,
                                    vtype=GRB.CONTINUOUS)
                    for t in range(self.horizon+1)]

        for t, (r_var, s_var, m_var) in enumerate(zip(r_vars, s_vars, m_vars)):
            self.model.addConstr(s_var == (self.threshold - signal[t]))
            self.model.addConstr(m_var == self.M * (1 - 2 * self.indicator[t]))
            self.model.addConstr(r_var == grb.max_(s_var, m_var))

        self.model.addConstr(rho == grb.min_(r_vars))
        return rho


    def impurity_optimization(self, rho, pos_indices, neg_indices):
        S_true = [self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS) for t in range(self.num_signals)]
        S_false = [self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS) for t in range(self.num_signals)]
        MR_true = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)
        MR_false = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)
        obj = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)

        z_true = [self.model.addVar(vtype = GRB.BINARY) for t in range(self.num_signals)]
        z_true_pos = [self.model.addVar(vtype = GRB.BINARY) for t in range(self.num_signals)]
        z_true_neg = [self.model.addVar(vtype = GRB.BINARY) for t in range(self.num_signals)]
        z_true_pos_cardinality = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)
        z_true_neg_cardinality = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)

        z_false_pos = [self.model.addVar(vtype = GRB.BINARY) for t in range(self.num_signals)]
        z_false_neg = [self.model.addVar(vtype = GRB.BINARY) for t in range(self.num_signals)]
        z_false_pos_cardinality = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)
        z_false_neg_cardinality = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS)
        
        ################ True ones ######################
        for i in range(self.num_signals):
            self.model.addConstr(S_true[i] == grb.max_(rho[i], 0))
            self.model.addGenConstrIndicator(z_true[i], True, S_true[i] >= 0.0001)
            self.model.addGenConstrIndicator(z_true[i], False, S_true[i] <= 0.0)

        ## True positive
        for i in neg_indices:
            self.model.addConstr(z_true_pos[i] == 0)

        for i in pos_indices:
            self.model.addGenConstrIndicator(z_true_pos[i], True, z_true[i] >= 0.1)
            self.model.addGenConstrIndicator(z_true_pos[i], False, z_true[i] <= 0.0)

        ## True negative
        for i in pos_indices:
            self.model.addConstr(z_true_neg[i] == 0)

        for i in neg_indices:
            self.model.addGenConstrIndicator(z_true_neg[i], True, z_true[i] >= 0.1)
            self.model.addGenConstrIndicator(z_true_neg[i], False, z_true[i] <= 0.0)

        self.model.addConstr(z_true_pos_cardinality == grb.quicksum(z_true_pos))
        self.model.addConstr(z_true_neg_cardinality == grb.quicksum(z_true_neg))
        self.model.addConstr(MR_true == grb.min_(z_true_pos_cardinality, z_true_neg_cardinality))


        ################# Negative ones ####################
        ## False positive
        for i in neg_indices:
            self.model.addConstr(z_false_pos[i] == 0)

        for i in pos_indices:
            self.model.addGenConstrIndicator(z_false_pos[i], True, z_true[i] <= 0.0)
            self.model.addGenConstrIndicator(z_false_pos[i], False, z_true[i] >= 0.1)

        ## False negative
        for i in pos_indices:
            self.model.addConstr(z_false_neg[i] == 0)

        for i in neg_indices:
            self.model.addGenConstrIndicator(z_false_neg[i], True, z_true[i] <= 0.0)
            self.model.addGenConstrIndicator(z_false_neg[i], False, z_true[i] >= 0.1)

        self.model.addConstr(z_false_pos_cardinality == grb.quicksum(z_false_pos))
        self.model.addConstr(z_false_neg_cardinality == grb.quicksum(z_false_neg))
        self.model.addConstr(MR_false == grb.min_(z_false_pos_cardinality, z_false_neg_cardinality))

        ################# Objective function #################
        self.model.addConstr(obj == grb.min_(MR_true, MR_false))
        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.update()

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
    num_signals     = len(labels)
    signals         = [mat_data['data'][i][0] for i in range(num_signals)]

    milp = PrimitiveMILP(signals, None)
    rho = [milp.predicate_robustness(i) for i in range(len(signals))]
    pos_indices, neg_indices = [], []
    for i in range(len(labels)):
        if labels[i] >= 0:
            pos_indices.append(i)
        else:
            neg_indices.append(i)
    milp.impurity_optimization(rho, pos_indices, neg_indices)

    

    



    # milp.model.setObjective(sum(rho[i] for i in pos_indices) - sum(rho[i] for i in neg_indices) + np.mean(milp.indicator) - milp.M * sum(epsilon), GRB.MAXIMIZE)
    # milp.model.setObjective(grb.quicksum(rho[i] for i in pos_indices) - grb.quicksum(rho[i] for i in neg_indices) + sum(milp.indicator)/len(milp.indicator), GRB.MAXIMIZE)

    # milp.model.update()
    milp.model.optimize()
    print(milp.model.status)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\tSolution:')
    print([(r.varName, r.x) for r in rho])
    print(milp.threshold.varName, milp.threshold.x)
    # print([var.x for var in milp.indicator])
    # print('Time interval', milp.get_interval())



def main():
    test1()


if __name__ == '__main__':
    main()
