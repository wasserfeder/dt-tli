import numpy as np
import random
from numpy import linalg as LA
from stl_prim import set_stl1_pars, set_stl2_pars, set_combined_stl_pars
import copy
# from combined_pso_test import get_indices


# def compute_robustness(trace, params, primitive, primitive_type, rho_path):
#     primitive = set_combined_stl_pars(primitive, primitive_type, params)
#     rho_primitive = primitive.robustness(trace, 0)
#     return np.min([rho_primitive, rho_path])

def get_indices(primitive, primitive_type):
    if primitive_type == 3 or primitive_type == 4:
        children = primitive.child.children
        signal_indices = [int(children[i].variable.split("_")[1]) for i in range(len(children))]
    elif primitive_type == 5:
        signal_indices = [int(primitive.child.left.variable.split("_")[1]) , int(primitive.child.right.variable.split("_")[1])]

    return signal_indices


def compute_combined_robustness(signals, position, primitive, primitive_type, rho_path):
    rhos = [0] * len(signals)
    if primitive_type == 3 or primitive_type == 4:
        indices = get_indices(primitive, primitive_type)
        children = primitive.child.children
        t0, t1 = int(position[-2]), int(position[-1])
        for i in range(len(signals)):
            rho_primitive = []
            for t in range(t0, t1+1):
                rho_predicates = [0] * len(indices)
                for k in range(len(indices)):
                    if children[k].relation == 2:
                        rho_predicates[k] = position[k] - signals[i][indices[k]][t]
                    else:
                        rho_predicates[k] = signals[i][indices[k]][t] - position[k]
                rho_primitive.append(np.min(rho_predicates))
            if primitive_type == 3:
                rhos[i] = min(max(rho_primitive),rho_path[i])
            else:
                rhos[i] = min(min(rho_primitive), rho_path[i])
        return rhos

    elif primitive_type == 5:
        rhos = [0] * len(signals)
        indices = get_indices(primitive, primitive_type)
        left_child = primitive.child.left
        right_child = primitive.child.right
        t0, t1, t3 = int(position[2]), int(position[3]), int(position[4])
        for i in range(len(signals)):
            rho_primitive = []
            for t in range(t0, t1+1):
                rho_until = []
                for t_prime in range(t+1, t+t3-1):
                    rho_left = []
                    for t_zeta in range(t, t_prime):
                        if left_child.relation == 2:
                            rho_left.append(position[0] - signals[i][indices[0]][t_zeta])
                        else:
                            rho_left.append(signals[i][indices[0]][t_zeta])
                    left_rho = min(rho_left)
                    if right_child.relation == 2:
                        right_rho = position[1] - signals[i][indices[1]][t_prime]
                    else:
                        right_rho = signals[i][indices[1]][t_prime] - position[1]
                    rho_until.append(min(left_rho, right_rho))
                rho_primitive.append(max(rho_until))
            rhos[i] = min(max(rho_primitive), rho_path[i])
        return rhos



def pso_costFunc(position, signals, traces, labels, primitive, primitive_type, rho_path, D_t):
    if primitive_type == 3 or primitive_type == 4:
        t0, t1 = position[-2], position[-1]
        if t0 > t1:
            print("Wrong Input")
            return
    elif primitive_type == 5:
        t0, t1, t3 = position[2], position[3], position[4]
        signal_horizon = len(signals[0][0])-1
        if t0 > t1 or t1+t3 > signal_horizon:
            print("Wrong Input")
            return
    S_true, S_false = [], []
    S_true_pos, S_true_neg = [], []
    S_false_pos, S_false_neg = [], []

    primitive = copy.deepcopy(primitive)
    # rhos = [compute_robustness(traces[i], position, primitive, primitive_type, rho_path[i]) for i in range(len(signals))]
    rhos = compute_combined_robustness(signals, position, primitive, primitive_type, rho_path)
    for i in range(len(signals)):
        if rhos[i] >= 0:
            S_true.append(i)
        else:
            S_false.append(i)

    for i in S_true:
        if labels[i] > 0:
            # S_true_pos.append(D_t[i] * rhos[i])
            # S_true_pos.append(D_t[i])
            S_true_pos.append(1)
        else:
            # S_true_neg.append(D_t[i] * rhos[i])
            # S_true_neg.append(D_t[i])
            S_true_neg.append(1)

    for i in S_false:
        if labels[i] > 0:
            # S_false_pos.append(-D_t[i] * rhos[i])
            # S_false_pos.append(D_t[i])
            S_false_pos.append(1)
        else:
            # S_false_neg.append(-D_t[i] * rhos[i])
            # S_false_neg.append(D_t[i])
            S_false_neg.append(1)

    S_tp, S_tn = sum(S_true_pos), sum(S_true_neg)
    S_fp, S_fn = sum(S_false_pos), sum(S_false_neg)
    MR_true = min(S_tp, S_tn)
    MR_false = min(S_fp, S_fn)
    obj = MR_true + MR_false

    return obj


class Particle():
    def __init__(self, x0, v0, signals, traces, labels, bounds, primitive, primitive_type):
        self.position       = x0
        self.velocity       = v0
        self.signals        = signals
        self.traces         = traces
        self.labels         = labels
        self.bounds         = bounds
        self.primitive      = primitive
        self.primitive_type = primitive_type
        self.err_best_i     = None
        self.pos_best_i     = []


    def evaluate(self, costFunc, rho_path, D_t):
        self.err_i = costFunc(self.position, self.signals, self.traces, self.labels, self.primitive, self.primitive_type, rho_path, D_t)

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
        if self.primitive_type == 3 or self.primitive_type == 4:
            children = self.primitive.child.children
            for j in range(len(children)):
                self.velocity[j] = max(self.velocity[j], -10)
                self.velocity[j] = min(self.velocity[j], 10)
            self.velocity[-2] = max(self.velocity[-2], -5)
            self.velocity[-2] = min(self.velocity[-2], 5)
            self.velocity[-1] = max(self.velocity[-1], -5)
            self.velocity[-1] = min(self.velocity[-1], 5)

        elif self.primitive_type == 5:
            self.velocity[0] = max(self.velocity[0], -10)
            self.velocity[0] = min(self.velocity[0], 10)
            self.velocity[1] = max(self.velocity[1], -10)
            self.velocity[1] = min(self.velocity[1], 10)
            self.velocity[2] = max(self.velocity[2], -5)
            self.velocity[2] = min(self.velocity[2], 5)
            self.velocity[3] = max(self.velocity[3], -5)
            self.velocity[3] = min(self.velocity[3], 5)
            self.velocity[4] = max(self.velocity[4], 4)
            self.velocity[4] = min(self.velocity[4], -4)


    def update_position(self):
        self.position = self.position + self.velocity
        if self.primitive_type == 3 or self.primitive_type == 4:
            self.position[-2] = int(np.floor(self.position[-2]))-1
            self.position[-1] = int(np.round(self.position[-1]))+1
            self.position[-2] = max(self.position[-2], 0)
            self.position[-2] = min(self.position[-2], self.bounds[-1]-1)
            self.position[-1] = max(self.position[-1], 1)
            self.position[-1] = min(self.position[-1], self.bounds[-1])
            if self.position[-2] > self.position[-1]:
                temp = self.position[-2]
                self.position[-2] = self.position[-1]
                self.position[-1] = temp
            children = self.primitive.child.children
            for j in range(len(children)):
                self.position[j] = max(self.position[j], self.bounds[2*j])
                self.position[j] = min(self.position[j], self.bounds[2*j+1])

        elif self.primitive_type == 5:
            self.position[0] = max(self.position[0], self.bounds[0])
            self.position[0] = min(self.position[0], self.bounds[1])
            self.position[1] = max(self.position[1], self.bounds[2])
            self.position[1] = min(self.position[1], self.bounds[3])
            self.position[2] = int(np.floor(self.position[2]))-1
            self.position[3] = int(np.round(self.position[3]))+1
            self.position[4] = int(np.round(self.position[4]))+1
            self.position[2] = max(self.position[2], 0)
            self.position[3] = max(self.position[3], 1)
            self.position[4] = max(self.position[4], 1)
            self.position[4] = min(self.position[4], self.bounds[-1]-1)
            self.position[3] = min(self.position[3], self.bounds[-1] - self.position[4])
            self.position[2] = min(self.position[2], self.position[3] - 1)



class Combined_PSO():
    def __init__(self, signals, traces, labels, bounds, primitive, primitive_type, args):
        self.k_max              = args.k_max
        self.num_particles      = args.num_particles
        self.signals            = signals
        self.traces             = traces
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
        if self.primitive_type == 3 or self.primitive_type == 4:
            children = self.primitive.child.children
            for i in range(self.num_particles):
                x0, v0 = [], []
                for j in range(len(children)):
                    pi_init = random.uniform(self.bounds[2*j], self.bounds[2*j+1])
                    x0 += [pi_init]
                    pi_range = (self.bounds[2*j+1] - self.bounds[2*j]) / self.num_particles
                    v0_pi = random.uniform(-pi_range, pi_range)
                    v0 += [v0_pi]
                t0_init = int(np.floor(random.uniform(0, self.bounds[-1]-1)))
                t1_init = int(np.round(random.uniform(t0_init + 1, self.bounds[-1])))
                x0 += [t0_init, t1_init]
                v0_t0 = random.randint(-3,3)
                v0_t1 = random.randint(-3,3)
                v0 += [v0_t0, v0_t1]
                x0, v0 = np.array(x0), np.array(v0)
                swarm.append(Particle(x0, v0, self.signals, self.traces, self.labels, self.bounds, self.primitive, self.primitive_type))
        elif self.primitive_type == 5:
            for i in range(self.num_particles):
                x0, v0 = [], []
                left_pi_init = random.uniform(self.bounds[0], self.bounds[1])
                right_pi_init = random.uniform(self.bounds[2], self.bounds[3])
                t3_init = int(np.floor(random.uniform(1, self.bounds[-1]-1)))
                t1_up = self.bounds[-1] - t3_init
                t0_init = int(np.floor(random.uniform(0, t1_up-1)))
                t1_init = int(np.round(random.uniform(t0_init + 1, t1_up)))
                x0 = np.array([left_pi_init, right_pi_init, t0_init, t1_init, t3_init])
                left_pi_range = (self.bounds[1] - self.bounds[0])/self.num_particles
                right_pi_range = (self.bounds[3]-self.bounds[2])/self.num_particles
                v0_left_pi = random.uniform(-left_pi_range, left_pi_range)
                v0_right_pi = random.uniform(-right_pi_range, right_pi_range)
                v0_t0 = random.randint(-3,3)
                v0_t1 = random.randint(-3,3)
                v0_t3 = random.randint(-2, 2)
                v0 = np.array([v0_left_pi, v0_right_pi, v0_t0, v0_t1, v0_t3])
                swarm.append(Particle(x0, v0, self.signals, self.traces, self.labels, self.bounds, self.primitive, self.primitive_type))
        return swarm


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

                # if stop:
            for i in range(self.num_particles):
                self.swarm[i].update_velocity(self.pos_best_g)
                self.swarm[i].update_position()

        return self.pos_best_g, self.err_best_g
