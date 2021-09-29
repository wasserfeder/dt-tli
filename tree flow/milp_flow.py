#!/usr/bin/env python

'''TODO:
The code is based on the MILP formulations of https://github.com/wasserfeder/dt-tli/tree/stlinf and https://github.com/pashew94/StrongTree
'''

import os
import argparse
import math
import time
from utils.Tree import Tree
import gurobipy as grb
from gurobipy import *
import numpy as np
from scipy.io import loadmat, savemat
import pickle
from robustness import trace_robustness_order1_lkt_1d
from robustness import trace_robustness_order1_lkt_nd
from robustness import traces_robustness_order1_lkt
from stl_prim import make_stl_primitives1, make_stl_primitives2


class FlowMILP(object):
    '''TODO:
    '''

    def __init__(self, signals, labels, ranges, rho_path, signal_dimension=0, depth=5, model=None):
        '''TODO:
        '''
        if model is not None:
            self.model = model
        else:
            self.model = grb.Model()

        self.variables = {}
        self.signals = signals
        print("signals:", len(self.signals[0]))
        self.labels = labels
        self.horizon = len(signals[0][0])
        self.num_signals = len(signals)
        self.rho_path = rho_path
        self.signal_dimension = signal_dimension
        self.depth = depth
        self.tree = Tree(self.depth)
        self._lambda = 0.8
        self.signal_index = list(range(0,self.num_signals))
        print("indices:", self.signal_index)

        primitives_list = make_stl_primitives1(self.signals)
        print('primitives_list:', primitives_list)
        self.primitive_index = list(range(0,len(primitives_list)))

        # Decision Variables
        '''
        self.b = 0   # binary, b[n] == 1 :if branching at node n [cannot be labelled]
        self.w = 0   # binary, w[n] == 1 :if label at node n [cannot branch]
        self.zeta = 0 # integer, zeta[n,t] ==  no of signals labelled and sent to sink
        self.z = 0 # integer, z[n][l(n)] == no of signals sent to the left child, z[n][r(n)] == no of signals sent to the right child
        '''
        self.b = 0
        self.w = 0
        self.zeta = 0
        self.z = 0
        self.z_phi = 0

        min_signals, max_signals = [], []
        for i in range(len(self.signals)):
            min_signals.append(min(signals[i][self.signal_dimension]))
            max_signals.append(max(signals[i][self.signal_dimension]))
        min_thresh = min(min_signals)
        max_thresh = max(max_signals)

        self.M = 12 # for SimpleDS

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
                             name='mutual_exclusive_time_intervals')   ##how does the constraint work?


        ###################### VARIABLES ###############################

        # b[n,f] ==1 iff at node n we branch on feature f
        self.b = self.model.addVars(self.tree.Nodes, vtype=GRB.BINARY, name='b')
        # p[n] == 1 iff at node n we do not branch and we make a prediction
        self.wp = self.model.addVars(self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name='wp')
        self.wn = self.model.addVars(self.tree.Nodes + self.tree.Leaves, vtype=GRB.BINARY, name='wn')
        # zeta[i,n]: the amount of flow through the edge connecting node n to sink node t for signal i

        maxlen= 200
        total_nodes = self.tree.Nodes + self.tree.Leaves
        sink = self.signal_index[-1]

        print('indices:', self.signal_index)

        self.zeta = self.model.addVars(self.signal_index, self.tree.Nodes + self.tree.Leaves, vtype=GRB.CONTINUOUS, lb=0,
                                       name='zeta')

        # z[i,n]: the incoming flow to node n for signal i
        # vname = 'z_{}_{}'.format(self.signal_index, self.tree.Nodes + self.tree.Leaves)
        self.z = self.model.addVars(self.signal_index, self.tree.Nodes + self.tree.Leaves, vtype=GRB.CONTINUOUS, lb=0,
                                    name='z')

        # self.z = self.model.addVars(vtype=GRB.CONTINUOUS, lb=0, name=vname)

        # z_phi[i,n] denotes whether or not signal i satisfies a primitive phi at node n
        # vname = 'z_{}_{}_{}'.format(self.signal_index, self.tree.Nodes + self.tree.Leaves , self.primitive_index)
        self.z_phi = self.model.addVars(self.signal_index, self.tree.Nodes+self.tree.Leaves, self.primitive_index, vtype=GRB.BINARY, name='z_phi')

        self.primitive_variables = [self.robustness(i, signal_dimension, np.inf)
                               for i in range(self.num_signals)]
        self.model.update()

        print("nodes:", self.tree.Nodes)
        print("leaves:", self.tree.Leaves)

        # x = [i for i in self.signals]
        # self.model.addConstrs((self.b[x]= 1) for x in self.tree.Nodes  )


        ######################## CONSTRAINTS #############################
        #
        # # Either branch or classify but not both
        self.model.addConstrs((self.b[n] + self.wp[n] + self.wn[n] == 1)  for n in self.tree.Nodes)
        #
        # Assign unique label
        self.model.addConstrs((self.wp[i] + self.wn[i] == 1)  for i in self.tree.Leaves)

        # Flow constraints
        # z[i,n]  = z[i, l(n)] + z[i, r(n)] + zeta[i, n] for all non-terminal nodes
        for current_node in self.tree.Nodes:
            # print("current:",current_node)
            predecessor = int(self.tree.get_parent(current_node))
            # print("predecessor;",predecessor)
            children_left = int(self.tree.get_left_children(current_node))
            # print("left:",children_left)
            children_right = int(self.tree.get_right_children(current_node))
            self.model.addConstrs(
                (self.z[i, current_node] == self.z[i, children_left] + self.z[i, children_right] + self.zeta[i, current_node]) for i in self.signal_index)

        # z[i,n]  = zeta[i, n] for all terminal nodes
        for current_node in self.tree.Leaves:
            # predecessor = int(self.tree.get_parent(current_node))
            self.model.addConstrs(
                (self.z[i, current_node] == self.zeta[i, current_node]) for i in self.signal_index)


        # A signal will enter only if it can reach the sink and should enter at max once
        self.model.addConstrs((self.z[i,1] <= 1) for i in self.signal_index)

        # if branched, should go to one of the children


        for current_node in self.tree.Nodes:
            children_left = int(self.tree.get_left_children(current_node))
            children_right = int(self.tree.get_right_children(current_node))
            self.model.addConstr(self.z[current_node,children_left] <= self.b[current_node])
            self.model.addConstr(self.z[current_node,children_right] <= self.b[current_node])


        for current_node in self.tree.Nodes + self.tree.Leaves:
            self.model.addConstrs((self.zeta[i, current_node] <= max(self.wp[current_node],self.wn[current_node])) for i in self.signal_index)

        #Robustness encoding

        if self.model.status == GRB.OPTIMAL:
            rho_sig = np.array([rho.X for rho in self.primitive_variables])
            # print("rho:", rho_sig[1])
            for i in self.signal_index:
                for current_node in self.tree.Nodes + self.tree.Leaves:
                    for j in self.primitive_index:
                        print("i", rho_sig[i])
                        self.model.addConstr(self.z_phi[i,current_node,j] * self.M >= rho_sig[i])
                        self.model.addConstr(-(1-self.z_phi[i,current_node,j]) * self.M <= rho_sig[i])


        # Ensure branching occurs iff a signal satisfies a particular primitive



        # define objective function
        obj = LinExpr(0)
        for i in self.signal_index:
            for current_node in self.tree.Nodes + self.tree.Leaves:
                obj.add((1 - self._lambda) * self.zeta[i, current_node])

        for n in self.tree.Nodes:
            obj.add(-1 * self._lambda * self.b[n])

        self.model.setObjective(obj, GRB.MAXIMIZE)

    def robustness(self, signal_index, signal_dimension=0, upper_bound=None):
        '''TODO:
        '''
        print("signal_index:",signal_index)
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
        self.model.addConstr(rho == grb.min_(primitive, upper_bound))   ## min, max, sum of python doesnt work -- use the one from gurobi

        vname = 'z_primitive_{}_{}'.format(signal_index, signal_dimension) ## interval values; only one z_primitive can be 1 for the entire dataset
        z = self.model.addVar(name=vname, vtype=GRB.BINARY)
        self.model.addConstr(rho <= z * self.M)
        self.model.addConstr(-rho <= (1 - z) * self.M)
        # return rho, z
        self.model.update()

        return rho

    def get_threshold(self):
        if self.model.status == GRB.OPTIMAL:
            return self.threshold.X
        else:
            raise RuntimeError('The model needs to be solved first!')
    #
    def get_robustnesses(self):
        if self.model.status == GRB.OPTIMAL:
            return np.array([rho.X for rho in self.primitive_variables])
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
    signals         = mat_data['data']      # alw >

    rho_path        = [np.inf for signal in signals]
    print("data:",signals)
    print('Number of signals:', len(signals))
    print('Time points:', len(timepoints))


    t0 = time.time()
    milp = FlowMILP(signals, labels, None, rho_path)
    milp.model.update()
    dt = time.time() - t0
    print('Setup time:', dt)

    t0 = time.time()
    milp.model.optimize()
    dt = time.time() - t0

    b_value = milp.model.getAttr("X", milp.b)
    p_value = milp.model.getAttr("X", milp.wp)

    print('Runtime:', dt)
    print('Model optimization status:', milp.model.status)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\tSolution:')
    # print('Threshold:', milp.get_threshold())
    # print('Time interval:', milp.get_interval())
    print('Objective value:', milp.model.objVal)
    # print("correctly classified signals:", count)
    # print('values:', milp.get_values())
    ####
    # sat_indices, unsat_indices = [], []
    # rho = milp.get_robustnesses()
    # for i in range(len(signals)):
    #     if rho[i] >= 0:
    #         sat_indices.append(i)
    #     else:
    #         unsat_indices.append(i)
    #
    # print("number of satisfying signals:", len(sat_indices))
    # print("number of violating signals:", len(unsat_indices))
    ####

def main():
    test1()

if __name__ == '__main__':
    main()
