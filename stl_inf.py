"""
Main inference module.

Contains the decision tree construction and related definitions.

Author: Erfan Aasi (eaasi@bu.edu)

"""
from stl_syntax import Formula, AND, OR, NOT, satisfies, robustness, GT
import numpy as np
from test_optimization_problem_sum_interval import PrimitiveMILP
from pso_test import get_bounds, run_pso_optimization
from pso import compute_robustness, PSO


class DTree(object):        # Decission tree recursive structure
    def __init__(self, primitive, signals, left=None, right=None):
        self.primitive = primitive
        self.signals = signals
        self.left = left
        self.right = right


    def classify(self, signal):
        if satisfies(self.primitive, SimpleModel(signal)):
            if self.left is None:
                return 1
            else:
                return self.left.classify(signal)
        else:
            if self.right is None:
                return -1
            else:
                return self.right.classify(signal)


    def get_formula(self):
        left = self.primitive
        right = Formula(NOT, [self.primitive])
        if self.left is not None:
            left = Formula(AND, [self.primitive, self.left.get_formula()])
        if self.right is not None:
            return Formula(OR, [left, Formula(AND, [right, self.right.get_formula()])])
        else:
            return left




def build_tree(signals, labels, timepoints, depth, primitives1, rho_path):
    # Check stopping conditions
    if depth <= 0:
        return None


    # primitive, impurity, robustnesses = find_best_primitive_milp(signals, labels, timepoints, primitives1, rho_path)
    primitive, impurity, robustnesses = find_best_primitive_pso(signals, labels, timepoints, primitives1, rho_path)
    print('Primitive:', primitive)
    print('impurity:', impurity)
    tree = DTree(primitive, signals)
    sat_signals = []
    sat_indices = []
    sat_rho = []
    sat_labels = []
    unsat_signals = []
    unsat_indices = []
    unsat_rho = []
    unsat_labels = []

    for i in range(len(signals)):
        if robustnesses[i] >= 0:
            sat_signals.append(signals[i])
            sat_rho.append(robustnesses[i])
            sat_labels.append(labels[i])
            sat_indices.append(i)
        else:
            unsat_signals.append(signals[i])
            unsat_rho.append(-robustnesses[i])
            unsat_labels.append(labels[i])
            unsat_indices.append(i)
    sat_signals = np.array(sat_signals)
    unsat_signals = np.array(unsat_signals)

    # if impurity <= 5:
    #     return None

    print("number of satisfying signals:", len(sat_signals))
    print("number of violating signals:", len(unsat_signals))

    # Recursively build the tree
    tree.left = build_tree(sat_signals, sat_labels, timepoints, depth - 1, primitives1, sat_rho)
    tree.right = build_tree(unsat_signals, unsat_labels, timepoints, depth - 1, primitives1, unsat_rho)

    return tree





def find_best_primitive_milp(signals, labels, timepoints, primitives1, rho_path):
    opt_prims = []
    for primitive in primitives1:
        primitive = primitive.copy()
        if primitive.args[0].args[0].op == GT:
            milp = PrimitiveMILP(signals, labels, None, rho_path)
        else:
            milp = PrimitiveMILP(-signals, labels, None, rho_path)

        milp.impurity_optimization(signal_dimension = primitive.args[0].args[0].index)
        milp.model.optimize()
        if primitive.args[0].args[0].op == GT:
            primitive.pi = milp.get_threshold()
        else:
            primitive.pi = - milp.get_threshold()
        primitive.t0 = timepoints[int(milp.get_interval()[0])]
        primitive.t1 = timepoints[int(milp.get_interval()[1])]
        rhos = milp.get_robustnesses()
        opt_prims.append([primitive, milp.model.objVal, rhos])

    return min(opt_prims, key=lambda x: x[1])




def find_best_primitive_pso(signals, labels, timepoints, primitives1, rho_path):
    opt_prims = []
    for primitive in primitives1:
        primitive = primitive.copy()
        print("candidate primitive:", primitive)
        signal_dimension = primitive.args[0].args[0].index
        if primitive.args[0].args[0].op == GT:
            params, impurity = run_pso_optimization(signals, labels, rho_path, signal_dimension)
        else:
            params, impurity = run_pso_optimization(-signals, labels, rho_path, signal_dimension)

        if primitive.args[0].args[0].op == GT:
            primitive.pi = params[0]
        else:
            primitive.pi = - params[0]
        primitive.t0 = timepoints[int(params[1])]
        primitive.t1 = timepoints[int(params[2])]
        if primitive.args[0].args[0].op == GT:
            rhos = [compute_robustness(signals[i], params[0], int(params[1]), int(params[2]), signal_dimension, rho_path[i]) for i in range(len(signals))]
        else:
            rhos = [compute_robustness(-signals[i], params[0], int(params[1]), int(params[2]), signal_dimension, rho_path[i]) for i in range(len(signals))]
        opt_prims.append([primitive, impurity, rhos])

    return min(opt_prims, key=lambda x: x[1])
