import numpy as np
import random
from numpy import linalg as LA
# from stl_syntax import GT, LE
import sys
sys.path.append("/home/erfan/iitchs/catl_planning/python-stl/stl")
from stl import STLFormula, Operation, RelOperation
import copy
from stl_prim import STL_Param_Setter


def compute_robustness(trace, primitive, rho_path):
    rho_primitive = primitive.robustness(trace, 0)
    return np.min([rho_primitive, rho_path])


def pso_costFunc(position, signals, traces, labels, primitive, rho_path, D_t):
    S_true, S_false = [], []
    S_true_pos, S_true_neg = [], []
    S_false_pos, S_false_neg = [], []

    primitive = copy.deepcopy(primitive)
    stl_param_setter = STL_Param_Setter()
    primitive = stl_param_setter.set_pars(primitive, position)
    rhos = [compute_robustness(traces[i], primitive, rho_path[i]) for i in range(len(signals))]
    for i in range(len(signals)):
        if rhos[i] >= 0:
            S_true.append(i)
        else:
            S_false.append(i)

    for i in S_true:
        if labels[i] > 0:
            S_true_pos.append(D_t[i])
        else:
            S_true_neg.append(D_t[i])

    for i in S_false:
        if labels[i] > 0:
            S_false_pos.append(D_t[i])
        else:
            S_false_neg.append(D_t[i])

    S_tp, S_tn = sum(S_true_pos), sum(S_true_neg)
    S_fp, S_fn = sum(S_false_pos), sum(S_false_neg)
    MR_true = min(S_tp, S_tn)
    MR_false = min(S_fp, S_fn)
    obj = MR_true + MR_false

    return obj


class Particle():
    def __init__(self, x0, v0, signals, traces, labels, bounds, variables, primitive):
        self.position = x0
        self.velocity = v0
        self.signals = signals
        self.traces = traces
        self.labels = labels
        self.bounds = bounds
        self.variables = variables
        self.primitive = primitive
        self.err_best_i = None
        self.pos_best_i = []


    def evaluate(self, costFunc, rho_path, D_t):
        self.err_i = costFunc(self.position, self.signals, self.traces, self.labels, self.primitive, rho_path, D_t)

        if self.err_i < self.err_best_i or self.err_best_i is None:
            self.err_best_i = self.err_i
            self.pos_best_i = self.position


    def update_velocity(self, pos_best_g):
        w   = 0.6
        c1  = 1
        c2  = 1.5
        r1, r2 = random.random(), random.random()

        vel_cog_diff    = self.pos_best_i - self.position
        vel_cognitive   = c1 * np.multiply(r1, vel_cog_diff)
        vel_soc_diff    = pos_best_g - self.position
        vel_social      = c2 * np.multiply(r2, vel_soc_diff)
        self.velocity   = w * self.velocity + vel_cognitive + vel_social

        t = 0
        while t < len(self.variables):
            if self.variables[t] == 't':
                self.velocity[t] = max(self.velocity[t], -5)
                self.velocity[t] = min(self.velocity[t], 5)
                self.velocity[t+1] = max(self.velocity[t+1], -5)
                self.velocity[t+1] = min(self.velocity[t+1], 5)
                t = t+2
            else:
                self.velocity[t] = max(self.velocity[t], -10)
                self.velocity[t] = min(self.velocity[t], 10)
                t = t+1


    def update_position(self):
        self.position = self.position + self.velocity
        self.pred_counter = 0
        t = 0
        while t < len(self.variables):
            if self.variables[t] == 't':
                self.position[t] = int(np.floor(self.position[t]))-1
                self.position[t+1] = int(np.round(self.position[t+1]))+1
                self.position[t] = max(self.position[t], 0)
                self.position[t] = min(self.position[t], self.bounds[-1]-1)
                self.position[t+1] = max(self.position[t+1], 1)
                self.position[t+1] = min(self.position[t+1], self.bounds[-1])
                if self.position[t] > self.position[t+1]:
                    temp = self.position[t]
                    self.position[t] = self.position[t+1]
                    self.position[t+1] = temp
                t = t+2
            else:
                self.position[t] = max(self.position[t], self.bounds[self.pred_counter])
                self.position[t] = min(self.position[t], self.bounds[self.pred_counter+1])
                self.pred_counter = self.pred_counter + 2
                t = t+1



class PSO():
    def __init__(self, signals, traces, labels, bounds, primitive, args):
        self.k_max              = args.k_max
        self.num_particles      = args.num_particles
        self.signals            = signals
        self.traces             = traces
        self.labels             = labels
        self.costFunc           = pso_costFunc
        self.bounds             = bounds
        self.primitive          = primitive

        self.err_best_g = None
        self.pos_best_g = []

        self.variables = self.find_variables(primitive, [])
        self.swarm = self.initialize_swarm()


    def find_variables(self, primitive, variables):
        if primitive.op == 6 or primitive.op == 7:
            variables.append('t')
            variables.append('t')
            variables = self.find_variables(primitive.child, variables)
            return variables
        elif primitive.op == 3:
            for k in range(len(primitive.children)):
                variables = self.find_variables(primitive.children[k], variables)
            return variables
        elif primitive.op == 8:
            variables.append('pi')
            return variables
        elif primitive.op == 5:
            variables = self.find_variables(primitive.left, variables)
            variables.append('t')
            variables.append('t')
            variables = self.find_variables(primitive.right, variables)
            return variables




    def initialize_swarm(self):
        swarm = []
        for i in range(self.num_particles):
            self.pred_counter = 0
            x0, v0 = self.initialize_particle(self.primitive, [], [])
            x0, v0 = np.array(x0), np.array(v0)
            swarm.append(Particle(x0, v0, self.signals, self.traces, self.labels, self.bounds, self.variables, self.primitive))
        return swarm


    def initialize_particle(self, primitive, x0, v0):
        if primitive.op == 6 or primitive.op == 7:
            t0_init = int(np.floor(random.uniform(0, self.bounds[-1]-1)))
            t1_init = int(np.round(random.uniform(t0_init + 1, self.bounds[-1])))
            x0.append(t0_init)
            x0.append(t1_init)
            v0_t0 = random.randint(-3,3)
            v0_t1 = random.randint(-3,3)
            v0.append(v0_t0)
            v0.append(v0_t1)
            x0, v0 = self.initialize_particle(primitive.child, x0, v0)
            return x0, v0

        elif primitive.op == 3:
            for k in range(len(primitive.children)):
                x0, v0 = self.initialize_particle(primitive.children[k], x0, v0)
            return x0, v0

        elif primitive.op == 8:
            pi_init = random.uniform(self.bounds[self.pred_counter], self.bounds[self.pred_counter+1])
            pi_range = (self.bounds[self.pred_counter+1] - self.bounds[self.pred_counter]) / self.num_particles
            x0.append(pi_init)
            v0_pi = random.uniform(-pi_range, pi_range)
            v0.append(v0_pi)
            self.pred_counter = self.pred_counter + 2
            return x0, v0

        elif primitive.op == 5:
            x0, v0 = self.initialize_particle(primitive.left, x0, v0)
            t0_init = int(np.floor(random.uniform(0, self.bounds[-1]-1)))
            t1_init = int(np.round(random.uniform(t0_init + 1, self.bounds[-1])))
            x0.append(t0_init)
            x0.append(t1_init)
            v0_t0 = random.randint(-3,3)
            v0_t1 = random.randint(-3,3)
            v0.append(v0_t0)
            v0.append(v0_t1)
            x0, v0 = self.initialize_particle(primitive.right, x0, v0)
            return x0, v0



    def optimize_swarm(self, rho_path, D_t):
        for k in range(self.k_max):
            for i in range(self.num_particles):
                self.swarm[i].evaluate(self.costFunc, rho_path, D_t)

                if self.swarm[i].err_best_i < self.err_best_g or self.err_best_g is None:
                    self.err_best_g = self.swarm[i].err_best_i
                    self.pos_best_g = self.swarm[i].pos_best_i

            print("error_best_g:", self.err_best_g)
            print("pos_best_g:", self.pos_best_g)

            convergence = 0
            for i in range(self.num_particles):
                distance = self.pos_best_g - self.swarm[i].position
                convergence = convergence + LA.norm(distance)
            print("convergence:", convergence)

            for i in range(self.num_particles):
                self.swarm[i].update_velocity(self.pos_best_g)
                self.swarm[i].update_position()

        return self.pos_best_g, self.err_best_g
