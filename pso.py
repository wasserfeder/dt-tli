import numpy as np
import random


def robustness(signal, pi, t0, t1):
    rho = []
    for t in range(t0, t1+1):
        rho.append(signal[0][t]-pi)
    return np.min(rho)


def pso_costFunc(position, signals, labels):
    [pi, t0, t1] = position
    if t0 >= t1:
        print("Wrong Input")
        return
    S_true, S_false = [], []
    S_true_pos, S_true_neg = [], []
    S_false_pos, S_false_neg = [], []

    rhos = [robustness(signal, pi, int(t0), int(t1)) for signal in signals]
    for i in range(len(signals)):
        if rhos[i] >= 0:
            S_true.append(i)
        else:
            S_false.append(i)

    for i in S_true:
        if labels[i] > 0:
            S_true_pos.append(rhos[i])
        else:
            S_true_neg.append(rhos[i])

    for i in S_false:
        if labels[i] > 0:
            S_false_pos.append(-rhos[i])
        else:
            S_false_neg.append(-rhos[i])

    S_tp, S_tn = sum(S_true_pos), sum(S_true_neg)
    S_fp, S_fn = sum(S_false_pos), sum(S_false_neg)
    MR_true = min(S_tp, S_tn)
    MR_false = min(S_fp, S_fn)
    obj = MR_true + MR_false

    return obj


class Particle():
    def __init__(self, x0, v0, signals, labels, bounds):
        self.position = x0
        self.velocity = v0
        self.signals = signals
        self.labels = labels
        self.bounds = bounds
        self.err_best_i = None
        self.pos_best_i = []


    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position, self.signals, self.labels)

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


    def update_position(self):
        self.position = self.position + self.velocity
        self.position[1] = int(np.floor(self.position[1]))
        self.position[2] = int(np.round(self.position[2]))
        self.position[0] = max(self.position[0], self.bounds[0])
        self.position[0] = min(self.position[0], self.bounds[1])
        self.position[1] = max(self.position[1], 0)
        self.position[1] = min(self.position[1], self.bounds[2]-1)
        self.position[2] = max(self.position[2], 1)
        self.position[2] = min(self.position[2], self.bounds[2])
        if self.position[1] > self.position[2]:
            temp = self.position[1]
            self.position[1] = self.position[2]
            self.position[2] = temp
        ### bounding


class PSO():
    def __init__(self, signals, labels, bounds, signal_dimension):
        self.k_max              = 100       # max iterations
        self.num_particles      = 15        # number of particles
        self.signals            = signals
        self.labels             = labels
        self.costFunc           = pso_costFunc
        self.bounds             = bounds

        self.err_best_g = None
        self.pos_best_g = []

        # Initialize the swarm
        if True:
            self.swarm = self.initialize_swarm()
            return None

    def initialize_swarm(self):
        swarm = []
        pi_range = (self.bounds[1] - self.bounds[0]) / self.num_particles
        v0_pi = random.uniform(-pi_range, pi_range)
        v0_t0 = random.randint(-3,3)
        v0_t1 = random.randint(-3,3)
        for i in range(self.num_particles):
            pi_init = random.uniform(self.bounds[0], self.bounds[1])
            t0_init = int(np.floor(random.uniform(0, self.bounds[2]-1)))
            t1_init = int(np.round(random.uniform(t0_init + 1, self.bounds[2])))
            x0 = np.array([pi_init, t0_init, t1_init])
            v0 = np.array([v0_pi, v0_t0, v0_t1])
            swarm.append(Particle(x0, v0, self.signals, self.labels, self.bounds))
        return swarm


    def optimize_swarm(self):
        for k in range(self.k_max):
            for i in range(self.num_particles):
                self.swarm[i].evaluate(self.costFunc)

                if self.swarm[i].err_best_i < self.err_best_g or self.err_best_g is None:
                    self.err_best_g = self.swarm[i].err_best_i
                    self.pos_best_g = self.swarm[i].pos_best_i

            print("error_best_g:", self.err_best_g)
            print("pos_best_g:", self.pos_best_g)
                # if stop:
            for i in range(self.num_particles):
                self.swarm[i].update_velocity(self.pos_best_g)
                self.swarm[i].update_position()

        return [self.pos_best_g, self.err_best_g]
