import numpy as np
import math
import argparse
from os import path
import os
from combined_pso import Combined_PSO



def run_combined_pso(signals, traces, labels, rho_path, primitive, primitive_type, D_t, args):
    signal_indices = get_indices(primitive, primitive_type)
    bounds = get_bounds(signals, signal_indices)
    particle_swarm = Combined_PSO(signals, traces, labels, bounds, primitive, primitive_type, args)
    params, impurity = particle_swarm.optimize_swarm(rho_path, D_t)
    return params, impurity



def get_indices(primitive, primitive_type):
    if primitive_type == 3 or primitive_type == 4:
        children = primitive.child.children
        signal_indices = [int(children[i].variable.split("_")[1]) for i in range(len(children))]

    return signal_indices



def get_bounds(signals, signal_indices):
    bounds = []
    for k in signal_indices:
        min_signals = []
        max_signals = []
        for i in range(len(signals)):
            min_signals.append(min(signals[i][k]))
            max_signals.append(max(signals[i][k]))
        min_pi = min(min_signals)
        max_pi = max(max_signals)
        bounds += [min_pi, max_pi]
    max_t = len(signals[0][0]) - 1
    bounds += [max_t]
    return bounds
