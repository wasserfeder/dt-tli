
## Imports
import numpy as np
import gurobipy as grb
from gurobipy import GRB
from scipy.interpolate import interp1d
from stl_syntax import LE, GT


class Optimize_Misclass_Gain(object):
    def __init__(self, signals, primitive, prim_level, prev_rho, pdist, disp = False):

        self.M              = 100000
        self.epsilon        = 0.001
        self.signals        = signals
        self.traces         = signals.traces
        self.labels         = signals.labels
        self.pos_indices    = signals.pos_indices
        self.neg_indices    = signals.neg_indices
        self.timepoints     = signals.timepoints
        self.N              = len(self.labels)
        self.T              = len(self.timepoints)
        self.pdist          = pdist
        self.primitive      = primitive
        self.prim_level     = prim_level
        self.prev_rho       = prev_rho
        if self.prev_rho is None:
            self.prev_rho   = [10000 for i in self.labels]
        self.model          = grb.Model()
        # self.model.setParam('OutputFlag', False)
        self.formulate_optimization()


    def formulate_optimization(self):
        self.add_variables()
        self.add_init_constraints()
        self.add_pos_class_constraints()
        self.add_neg_class_constraints()
        self.add_true_pos_constraints()
        self.add_true_neg_constraints()
        self.add_primitive_constraints()
        self.add_objective_function()


    def add_variables(self):
        self.u = self.model.addVar(name='u',lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                                        vtype=GRB.CONTINUOUS)
        self.v1 = self.model.addVar(name='v1',lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                                        vtype=GRB.CONTINUOUS)
        self.v2 = self.model.addVar(name='v2',lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                                        vtype=GRB.CONTINUOUS)
        self.v3 = self.model.addVar(name='v3',lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                                        vtype=GRB.CONTINUOUS)
        self.z3 = self.model.addVar(vtype=GRB.BINARY)
        self.vr_p = self.model.addVar(name='vr_p',lb=-GRB.INFINITY,
                                        ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        self.vr_n = self.model.addVar(name='vr_n',lb=-GRB.INFINITY,
                                        ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        self.vr_tp = self.model.addVar(name='vr_tp',lb=-GRB.INFINITY,
                                        ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        self.vr_tn = self.model.addVar(name='vr_tn',lb=-GRB.INFINITY,
                                        ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)

        self.vr_prim = [self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS) for t in range(self.N)]
        self.vr_pn = [self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS) for t in range(self.N)]
        self.abs_vr_pn = [self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS) for t in range(self.N)]
        self.vr_tpn = [self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS) for t in range(self.N)]
        self.vr_max = [self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS) for t in range(self.N)]
        self.model.update()


    def add_init_constraints(self):
        self.model.addConstr(self.v1 == grb.min_(self.vr_p, self.vr_n))
        self.model.addConstr(self.v2 == grb.min_(self.vr_tp, self.vr_tn))
        self.model.addConstr(self.v3 <= self.vr_p - self.vr_tp)
        self.model.addConstr(self.v3 <= self.vr_n - self.vr_tn)
        self.model.addConstr(self.v3 == self.z3 * (self.vr_p - self.vr_tp))
        self.model.addConstr(self.v3 == (1 - self.z3) * (self.vr_n - self.vr_tn))
        # self.model.addConstr(self.v3 == grb.min_(self.vr_p - self.vr_tp,
        #                                             self.vr_n - self.vr_tn))
        self.model.addConstr(self.vr_p + self.vr_n == 1)
        self.model.addConstr(self.u > 0)
        self.model.addConstr(self.v1 >= 0)
        self.model.addConstr(self.v2 >= 0)
        self.model.addConstr(self.v3 >= 0)
        self.model.addConstr(self.vr_p >= 0)
        self.model.addConstr(self.vr_n >= 0)
        self.model.addConstr(self.vr_tp >= 0)
        self.model.addConstr(self.vr_tn >= 0)
        self.model.update()


    def add_pos_class_constraints(self):
        for i in self.pos_indices:
            self.abs_vr_pn[i] = grb.abs_(self.vr_pn[i])
        self.model.addConstr(self.vr_p == sum(self.pdist[i] *
                            self.abs_vr_pn[i] for i in self.pos_indices))
        # self.model.addConstr(self.vr_p == sum(self.pdist[i] * grb.abs_(self.vr_pn[i]) for i in self.pos_indices))
        for i in self.pos_indices:
            self.model.addConstr(self.vr_pn[i] ==
                        grb.min_(self.u * self.prev_rho[i], self.vr_prim[i]))
        self.mode7l.update()


    def add_neg_class_constraints(self):
        self.model.addConstr(self.vr_n == sum(self.pdist[i] *
                            grb.abs_(self.vr_pn[i]) for i in self.neg_indices))
        for i in self.neg_indices:
            self.model.addConstr(self.vr_pn[i] ==
                        grb.min_(self.u * self.prev_rho[i], self.vr_prim[i]))
        self.model.update()
        

    def add_true_pos_constraints(self):
        self.model.addConstr(self.vr_tp == sum(self.pdist[i] *
                            grb.abs_(self.vr_tpn[i]) for i in self.pos_indices))
        for i in self.pos_indices:
            self.model.addConstr(self.vr_tpn[i] ==
                            grb.min_(self.u * self.prev_rho[i], self.vr_max[i]))
            self.model.addConstr(self.vr_max[i] ==
                            grb.max_(self.vr_prim[i], self.u * self.epsilon))
        self.model.update()


    def add_true_neg_constraints(self):
        self.model.addConstr(self.vr_tn == sum(self.pdist[i] *
                            grb.abs_(self.vr_tpn[i]) for i in self.neg_indices))
        for i in self.neg_indices:
            self.model.addConstr(self.vr_tpn[i] ==
                            grb.min_(self.u * self.prev_rho[i], self.vr_max[i]))
            self.model.addConstr(self.vr_max[i] ==
                            grb.max_(self.vr_prim[i], self.u * self.epsilon))
        self.model.update()


    def add_primitive_constraints(self):
        self.min_t = self.timepoints[0]
        self.max_t = self.timepoints[-1]
        self.min_pi = np.min(self.signals.traces)
        self.max_pi = np.max(self.signals.traces)
        self.model.update()

        if self.prim_level == 1:
            self.add_first_level_variables()
            self.add_first_level_constraints()
        # else:
        #     self.add_second_level_variables()
        #     self.add_second_level_constraints()
            # out_expr = self.primitive._op
            # if out_expr == 5:   # always(eventually) primitive
            #     self.add_second_always_constraints()
            # else:               # eventually(always) primitive
            #     self.add_second_eventually_constraints()

    def add_first_level_variables(self):
        self.vpi = self.model.addVar(name='u',lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                                        vtype=GRB.CONTINUOUS)
        self.vr_prim_max = [[self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS) for t in range(self.N)] for j in range(self.T)]
        self.ind = [self.model.addVar(vtype=GRB.BINARY) for j in range(self.T)]
        self.ind_inc = [self.model.addVar(vtype=GRB.BINARY) for j in range(self.T)]
        self.ind_dec = [self.model.addVar(vtype=GRB.BINARY) for j in range(self.T)]
        self.zind = [self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS) for j in range(self.T)]
        self.zind_inc = [self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS) for j in range(self.T)]
        self.zind_dec = [self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,
                            vtype=GRB.CONTINUOUS) for j in range(self.T)]
        self.model.update()

    def add_first_level_constraints(self):
        expr    = self.primitive._op   
        index   = self.primitive.index                  # index of the signal
        op      = self.primitive.args[0].args[0].op     # operator of the predicate
        for i in range(self.N):
            if expr == 5:   # always operator
                self.model.addConstr(self.vr_prim[i] == grb.min_(self.vr_prim_max[i]))
            else:           # eventually operator
                self.model.addConstr(self.vr_prim[i] == grb.max_(self.vr_prim_max[i]))

            for t in range(self.T):
                if op == LE:
                    self.model.addConstr(self.vr_prim_max[i][t] == grb.max_(self.vpi - self.u * self.traces[i][index][t], self.M * (self.u - 2 * self.zind[t])))
                else:
                    self.model.addConstr(self.vr_prim_max[i][t] == grb.max_(self.u * self.traces[i][index][t] - self.vpi, self.M * (self.u - 2 * self.zind[t])))

        ### Equation (44)
        for t in range(self.T):
            self.model.addConstr(self.zind[t] == grb.min_(self.zind_inc[t], self.zind_dec[t]))

        for t in range(self.T-1):
            self.model.addConstr(self.zind_inc[t] <= self.zind_inc[t+1])
            self.model.addConstr(self.zind_dec[t] >= self.zind_dec[t+1])
        
        self.model.addConstr(sum(self.zind) > 0)

        ### Equation (45)
        for t in range(self.T): 
            self.model.addConstr(self.zind[t] <= self.u)
            self.model.addConstr(self.zind[t] <= self.M * self.ind[t])
            self.model.addConstr(self.zind[t] >= self.u - self.M * (1 - self.ind[t]))
            self.model.addConstr(self.zind[t] >= 0)

            self.model.addConstr(self.zind_inc[t] <= self.u)
            self.model.addConstr(self.zind_inc[t] <= self.M * self.ind_inc[t])
            self.model.addConstr(self.zind_inc[t] >= self.u - self.M * (1 - self.ind_inc[t]))
            self.model.addConstr(self.zind_inc[t] >= 0)

            self.model.addConstr(self.zind_dec[t] <= self.u)
            self.model.addConstr(self.zind_dec[t] <= self.M * self.ind_dec[t])
            self.model.addConstr(self.zind_dec[t] >= self.u - self.M * (1 - self.ind_dec[t]))
            self.model.addConstr(self.zind_dec[t] >= 0)
        self.model.update()


    # def add_second_level_variables(self):


    
    # def add_second_level_constraints(self):


    def add_objective_function(self):
        self.model.setObjective(self.v1 - self.v2 - self.v3, GRB.MAXIMIZE)
        self.model.update()


    def get_solution(self):
        self.model.optimize()
        print(self.model.status)
        # m.computeIIS()
        # m.write("model.ilp")
        # m.write("model.mps")
        if self.model.status == 2:
            self.primitive.pi = self.vpi.X/self.u.X
            print(self.primitive.pi)
            print(self.ind_inc)
            print(self.ind_dec)
            for t in range(self.T-1):
                if (self.ind_inc[t].X < 0.5) and (self.ind_inc[t+1].X >= 0.5):
                # if not id_t_inc[t].X and id_t_inc[t+1].X:
                # if not w_id_t_inc[t].X/u.X and w_id_t_inc[t+1].X/u.X:
                    t0 = t+1
                    print(t0)
                    # break
            for t in range(self.T-1):
                if (self.ind_dec[t].X >= 0.5) and (self.ind_dec[t+1].X < 0.5):
                # if id_t_dec[t].X and not id_t_dec[t+1].X:
                # if w_id_t_dec[t].X/u.X and not w_id_t_dec[t+1].X/u.X:
                    t1 = t
                    print(t1)
                    # break
                # primitive.t0 = t0
                # primitive.t1 = t1

            return self.primitive, self.model.getObjective().getValue()
        else:
            return None, None





    # Second-level primitives
    # else:
    #     t1 = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    #     t2 = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    #     t3 = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    #     pi = m.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY)
    #     out_expr = primitive._op
    #     in_expr = primitive.args[0]._op
    #     index = primitive.args[0].args[0].args[0].index
    #     op = primitive.args[0].args[0].args[0].op

        # always(eventually) operator
        # if expr == 5:

        # eventually(always) operator
        # else:
