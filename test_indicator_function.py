#!/usr/bin/env python

'''TODO:
'''

import gurobipy as grb
from gurobipy import GRB
import numpy as np


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

        self.rho_max = 2 # TODO: set proper value
        self.M = 3 # TODO: set proper value
        self.horizon = len(self.signals[0]) - 1

        # primitive threshold
        self.threshold = self.model.addVar(name='threshold',
                                           lb=-self.rho_max, ub=self.rho_max,
                                           vtype=GRB.CONTINUOUS)
        # primitive time interval indicator function
        self.indicator = self.indicator_function()

    # def min(self, min_var, args, custom=False):
    #     '''
    #     grb.min_(ind_inc[t], ind_dec[t])
    #
    #     or custom
    #     '''
    #     if custom:
    #         z_vars = [self.model.addVar(vtype=GRB.BINARY)
    #                   for t in range(len(args))]
    #         for arg, z_var in zip(args, z_vars):
    #             self.model.addConstr(min_var <= arg)
    #             self.model.addConstr(min_var >= arg - self.M * (1 - z_var))
    #         self.model.addConstr(sum(z_vars) >= 1)
    #     else:
    #         vars = [self.model.addVar(vtype=arg.vtype, lb=arg.lb, ub=arg.ub)
    #                 for arg in args]
    #         self.model.addConstr(min_var == grb.min_(vars))
    #
    # def max(self, max_var, args, custom=False):
    #     '''
    #     '''
    #     if custom:
    #         z_vars = [self.model.addVar(vtype=GRB.BINARY)
    #                   for t in range(len(args))]
    #         for arg, z_var in zip(args, z_vars):
    #             self.model.addConstr(max_var >= arg)
    #             self.model.addConstr(min_var <= arg + self.M * (1 - z_var))
    #         self.model.addConstr(sum(z_vars) >= 1)
    #     else:
    #         vars = [self.model.addVar(vtype=arg.vtype, lb=arg.lb, ub=arg.ub)
    #                 for arg in args]
    #         self.model.addConstr(max_var == grb.max_(vars))

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

        if not custom_encoding:
            s_vars = [self.model.addVar(lb=-2, ub=2, vtype=GRB.CONTINUOUS)
                      for t in range(self.horizon+1)]
            m_vars = [self.model.addVar(lb=-self.M, ub=self.M,
                                        vtype=GRB.CONTINUOUS)
                      for t in range(self.horizon+1)]

            for t, (r_var, s_var, m_var) in enumerate(zip(r_vars, s_vars, m_vars)):
                self.model.addConstr(s_var == (signal[t] - self.threshold))
                self.model.addConstr(m_var == self.M * (1 - 2 * self.indicator[t]))
                self.model.addConstr(r_var == grb.max_(s_var, m_var))

            self.model.addConstr(rho == grb.min_(r_vars))

        else:
            for t, r_var in enumerate(r_vars):
                s_var = (signal[t] - self.threshold)
                m_var = self.M * (1 - 2 * self.indicator[t])
                self.model.addConstr(r_var >= s_var)
                self.model.addConstr(r_var >= m_var)
                z = self.model.addVar(vtype=GRB.BINARY)
                self.model.addConstr(r_var <= s_var + 10 * z)
                self.model.addConstr(r_var <= m_var + 10 * (1-z))

            z_vars = [self.model.addVar(vtype=GRB.BINARY)
                      for t in range(self.horizon+1)]
            for r_var, z_var in zip(r_vars, z_vars):
                self.model.addConstr(rho <= r_var)
                self.model.addConstr(rho >= r_var - self.M * (1 - z_var))
            self.model.addConstr(sum(z_vars) == 1)

        return rho

def test1():
    '''TODO:
    '''

    signals = [np.array([0.1, 0.2, 0.3]),
               np.array([0.1, 0.3, 0.4]),
               np.array([0.3, 0.1, 0.1])]

    milp = PrimitiveMILP(signals, None)
    rho = [milp.predicate_robustness(i) for i in range(len(signals))]
    milp.model.addConstr(rho[0] >= 0.01)
    milp.model.addConstr(rho[1] >= 0.01)
    milp.model.addConstr(rho[2] <= -0.01)

    # milp.model.setObjective(rho[0]+rho[1]-rho[2] + np.mean(milp.indicator), GRB.MAXIMIZE)
    milp.model.setObjective(rho[0]+rho[1]-rho[2], GRB.MAXIMIZE)
    milp.model.update()
    milp.model.optimize()
    print(milp.model.status)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\tSolution:')
    print([(r.varName, r.x) for r in rho])
    print(milp.threshold.varName, milp.threshold.x)
    print([var.x for var in milp.indicator])
    print('Time interval', milp.get_interval())

def main():
    test1()

if __name__ == '__main__':
    main()
