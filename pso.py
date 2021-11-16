import numpy as np
import random
from numpy import linalg as LA
from stl_prim import set_stl1_pars
import copy


def compute_robustness(signals, params, primitive, primitive_type, rho_paths):
    rhos = [0] * len(signals)
    if primitive_type == 1:
        index = int(primitive.child.variable.split("_")[1])
        threshold = params[0]
        t0, t1 = int(params[1]), int(params[2])
        if primitive.op == 6:
            if primitive.child.relation == 2:
                for i in range(len(signals)):
                    rhos[i] = np.min([np.max([threshold - signals[i][index][t] for t in range(t0, t1+1)]), rho_paths[i]])
            else:
                for i in range(len(signals)):
                    rhos[i] = np.min([np.max([signals[i][index][t] - threshold for t in range(t0, t1+1)]), rho_paths[i]])
        else:
            if primitive.child.relation == 2:
                for i in range(len(signals)):
                    rhos[i] = np.min([np.min([threshold - signals[i][index][t] for t in range(t0, t1+1)]), rho_paths[i]])
            else:
                for i in range(len(signals)):
                    rhos[i] = np.min([np.min([signals[i][index][t] - threshold for t in range(t0, t1+1)]), rho_paths[i]])
        return rhos



def pso_costFunc(position, signals, labels, primitive, primitive_type, rho_path):
    if primitive_type == 1:
        [pi, t0, t1] = position
        if t0 > t1:
            print("Wrong Input")
            return
    else:
        [pi, t0, t1, t3] = position
        signal_horizon = len(signals[0][0]) - 1
        if t0 > t1 or t1+ t3 > signal_horizon:
            print("Wrong Input")
            return
    S_true, S_false = [], []
    S_true_pos, S_true_neg = [], []
    S_false_pos, S_false_neg = [], []

    primitive = copy.deepcopy(primitive)
    rhos = compute_robustness(signals, position, primitive, primitive_type, rho_path)
    for i in range(len(signals)):
        if rhos[i] >= 0:
            S_true.append(i)
        else:
            S_false.append(i)

    for i in S_true:
        if labels[i] > 0:
            S_true_pos.append(1)
        else:
            S_true_neg.append(1)

    for i in S_false:
        if labels[i] > 0:
            S_false_pos.append(1)
        else:
            S_false_neg.append(1)

    S_tp, S_tn = sum(S_true_pos), sum(S_true_neg)
    S_fp, S_fn = sum(S_false_pos), sum(S_false_neg)
    MR_true = min(S_tp, S_tn)
    MR_false = min(S_fp, S_fn)
    obj = MR_true + MR_false

    return obj


class Particle():
    def __init__(self, x0, v0, signals,  labels, bounds, primitive, primitive_type):
        self.position       = x0
        self.velocity       = v0
        self.signals        = signals
        self.labels         = labels
        self.bounds         = bounds
        self.primitive      = primitive
        self.primitive_type = primitive_type
        self.err_best_i     = None
        self.pos_best_i     = []


    def evaluate(self, costFunc, rho_path):
        self.err_i = costFunc(self.position, self.signals, self.labels, self.primitive, self.primitive_type, rho_path)

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
        self.velocity[0] = max(self.velocity[0], -10)
        self.velocity[0] = min(self.velocity[0], 10)
        self.velocity[1] = max(self.velocity[1], -5)
        self.velocity[1] = min(self.velocity[1], 5)
        self.velocity[2] = max(self.velocity[2], -5)
        self.velocity[2] = min(self.velocity[2], 5)
        if self.primitive_type == 2:
            self.velocity[3] = max(self.velocity[3], 4)
            self.velocity[3] = min(self.velocity[3], -4)


    def update_position(self):
        self.position = self.position + self.velocity
        self.position[1] = int(np.floor(self.position[1]))-1
        self.position[2] = int(np.round(self.position[2]))+1
        self.position[0] = max(self.position[0], self.bounds[0])
        self.position[0] = min(self.position[0], self.bounds[1])
        if self.primitive_type == 1:
            self.position[1] = max(self.position[1], 0)
            self.position[1] = min(self.position[1], self.bounds[2]-1)
            self.position[2] = max(self.position[2], 1)
            self.position[2] = min(self.position[2], self.bounds[2])
            if self.position[1] > self.position[2]:
                temp = self.position[1]
                self.position[1] = self.position[2]
                self.position[2] = temp
        else:
            self.position[3] = int(np.round(self.position[3]))+1
            self.position[1] = max(self.position[1], 0)
            self.position[2] = max(self.position[2], 1)
            self.position[3] = max(self.position[3], 1)
            self.position[3] = min(self.position[3], self.bounds[2]-1)
            self.position[2] = min(self.position[2], self.bounds[2] - self.position[3])
            self.position[1] = min(self.position[1], self.position[2] - 1)


class PSO():
    def __init__(self, signals, labels, bounds, primitive, primitive_type, args):
        self.k_max              = args.k_max
        self.num_particles      = args.num_particles
        self.signals            = signals
        self.labels             = labels
        self.costFunc           = pso_costFunc
        self.bounds             = bounds
        self.primitive          = primitive
        self.primitive_type     = primitive_type

        self.err_best_g = None
        self.pos_best_g = []

        # Initialize the swarm
        self.swarm = self.initialize_swarm()


    def initialize_swarm(self):
        swarm = []
        pi_range = (self.bounds[1] - self.bounds[0]) / self.num_particles
        if self.primitive_type == 1:
            for i in range(self.num_particles):
                pi_init = random.uniform(self.bounds[0], self.bounds[1])
                t0_init = int(np.floor(random.uniform(0, self.bounds[2]-1)))
                t1_init = int(np.round(random.uniform(t0_init + 1, self.bounds[2])))
                x0 = np.array([pi_init, t0_init, t1_init])
                v0_pi = random.uniform(-pi_range, pi_range)
                v0_t0 = random.randint(-3,3)
                v0_t1 = random.randint(-3,3)
                v0 = np.array([v0_pi, v0_t0, v0_t1])
                swarm.append(Particle(x0, v0, self.signals, self.labels, self.bounds, self.primitive, self.primitive_type))
        else:
            for i in range(self.num_particles):
                pi_init = random.uniform(self.bounds[0], self.bounds[1])
                t3_init = int(np.floor(random.uniform(1, self.bounds[2]-1)))
                t1_up = self.bounds[2] - t3_init
                t0_init = int(np.floor(random.uniform(0, t1_up-1)))
                t1_init = int(np.round(random.uniform(t0_init + 1, t1_up)))
                x0 = np.array([pi_init, t0_init, t1_init, t3_init])
                v0_pi = random.uniform(-pi_range, pi_range)
                v0_t0 = random.randint(-3,3)
                v0_t1 = random.randint(-3,3)
                v0_t3 = random.randint(-2, 2)
                v0 = np.array([v0_pi, v0_t0, v0_t1, v0_t3])
                swarm.append(Particle(x0, v0, self.signals, self.labels, self.bounds, self.primitive, self.primitive_type))
        return swarm


    def optimize_swarm(self, rho_path):
        for k in range(self.k_max):
            for i in range(self.num_particles):
                self.swarm[i].evaluate(self.costFunc, rho_path)

                if self.swarm[i].err_best_i < self.err_best_g or self.err_best_g is None:
                    self.err_best_g = self.swarm[i].err_best_i
                    self.pos_best_g = self.swarm[i].pos_best_i

            # print("error_best_g:", self.err_best_g)
            # print("pos_best_g:", self.pos_best_g)

            convergence = 0
            for i in range(self.num_particles):
                distance = self.pos_best_g - self.swarm[i].position
                convergence = convergence + LA.norm(distance)
            # print("convergence:", convergence)

                # if stop:
            for i in range(self.num_particles):
                self.swarm[i].update_velocity(self.pos_best_g)
                self.swarm[i].update_position()

        return self.pos_best_g, self.err_best_g
