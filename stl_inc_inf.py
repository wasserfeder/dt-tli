
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

from stl_syntax import Formula, AND, OR, NOT, satisfies, robustness, GT
import numpy as np
from test_optimization_problem_sum_interval import PrimitiveMILP
from pso_test import get_bounds, run_pso_optimization
from pso import compute_robustness, PSO
from stl_prim import set_stl1_pars, set_stl2_pars
from scipy.stats import norm

# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

class DTree(object):        # Decission tree recursive structure
    def __init__(self, primitive, signals, left=None, right=None):
        self.primitive = primitive
        self.signals = signals
        self.left = left
        self.right = right


    def classify(self, signal):
        if satisfies(self.primitive, signal):
            if self.left is None:
                return 1
            else:
                return self.left.classify(signal)
        else:
            if self.right is None:
                return -1
            else:
                return self.right.classify(signal)

    def tree_robustness(self, signal, rho_path):
        pi = self.primitive.pi
        t0 = self.primitive.t0
        t1 = self.primitive.t1
        if self.primitive.type == 1:
            params = [pi, t0 ,t1]
        else:
            params = [pi, t0 ,t1, self.primitive.t3]
        rho = compute_robustness(signal, params, self.primitive, rho_path)
        if self.left is None:
            return rho
        elif rho >= 0 and self.left is not None:
            return self.left.tree_robustness(signal, rho)
        else:
            return self.right.tree_robustness(signal, -rho)





    def get_formula(self):
        left = self.primitive
        right = Formula(NOT, [self.primitive])
        if self.left is not None:
            left = Formula(AND, [self.primitive, self.left.get_formula()])
        if self.right is not None:
            return Formula(OR, [left, Formula(AND, [right, self.right.get_formula()])])
        else:
            return left


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================
def build_inc_tree(signals, labels, rho_path, depth, primitives, opt_type, D_t):
    # Check stopping conditions
    if (depth <= 0) or (len(signals) == 0):
        return None

    if opt_type == 'milp':
        prim, impurity, rhos = best_prim_inc_milp(signals, labels, rho_path, D_t,
                                                                    primitives)
    else:
        prim, impurity, rhos = best_prim_inc_pso(signals, labels, rho_path, D_t,
                                                                    primitives)

    if prim is None:
        return None
    # Check for EVENTUALLY primitives
    counter = 0
    reverse_counter = 0
    for i in range(len(signals)):
        if (rhos[i] >= 0 and labels[i] < 0) or (rhos[i] < 0 and labels[i] > 0):
            counter = counter + 1
        if (-rhos[i] >= 0 and labels[i] < 0) or (-rhos[i] < 0 and labels[i]> 0):
            reverse_counter = reverse_counter + 1

    if reverse_counter < counter:
        prim.reverse_rel()
        prim.reverse_op()
        rhos = - rhos

    print('***************************************************************')
    print('Primitive:', prim)
    print('impurity:', impurity)
    tree = DTree(prim, signals)
    sat_signals, unsat_signals  = [], []
    sat_indices, unsat_indices  = [], []
    sat_rho, unsat_rho          = [], []
    sat_labels, unsat_labels    = [], []
    sat_weights, unsat_weights  = [], []

    for i in range(len(signals)):
        if rhos[i] >= 0:
            sat_signals.append(signals[i])
            sat_rho.append(rhos[i])
            sat_labels.append(labels[i])
            sat_indices.append(i)
            sat_weights.append(D_t[i])

        else:
            unsat_signals.append(signals[i])
            unsat_rho.append(-rhos[i])
            unsat_labels.append(labels[i])
            unsat_indices.append(i)
            unsat_weights.append(D_t[i])

    sat_signals     = np.array(sat_signals)
    unsat_signals   = np.array(unsat_signals)

    # Recursively build the tree
    tree.left = build_inc_tree(sat_signals, sat_labels, sat_rho, depth - 1,
                primitives, opt_type, sat_weights)
    tree.right = build_inc_tree(unsat_signals, unsat_labels, unsat_rho, depth - 1,
                primitives, opt_type, unsat_weights)

    return tree


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def best_prim_inc_milp(signals, labels, rho_path, D_t, primitives):
    opt_prims = []
    for primitive in primitives:
        primitive = primitive.copy()
        print('***************************************************************')
        print("candidate primitive:", primitive)
        if primitive.rel == GT:
            milp = PrimitiveMILP(signals, labels, None, rho_path, D_t,
                   primitive.index)

        else:
            milp = PrimitiveMILP(-signals, labels, None, rho_path, D_t,
                   primitive.index)
        milp.impurity_optimization(signal_dimension = primitive.index)
        milp.model.optimize()
        pi          = milp.get_threshold()
        [t0, t1]    = milp.get_interval()
        if primitive.rel == GT:
            params = [pi, t0, t1]
        else:
            params = [-pi, t0, t1]
        # params = [pi, t0, t1]
        primitive   = set_stl1_pars(primitive, params)
        rhos        = milp.get_robustnesses()
        opt_prims.append([primitive, milp.model.objVal, rhos])

    return min(opt_prims, key=lambda x: x[1])


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def best_prim_inc_pso(signals, labels, rho_path, D_t, primitives):
    delta = 0.1
    quantile = norm.ppf(1-delta)
    epsilon = quantile * (1 / np.sqrt(2 * len(signals)))
    opt_prims = []
    for primitive in primitives:
        primitive = primitive.copy()
        print('***************************************************************')
        print("candidate primitive:", primitive)
        params, impurity = run_pso_optimization(signals, labels, rho_path, D_t,
                           primitive)
        if primitive.type == 1:
            primitive = set_stl1_pars(primitive, params)

        else:
            primitive = set_stl2_pars(primitive, params)

        rhos = np.array([compute_robustness(signals[i], params, primitive,
                                    rho_path[i]) for i in range(len(signals))])
        opt_prims.append([primitive, impurity, rhos])

    index = None
    for i in range(len(opt_prims)):
        imp = opt_prims[i][1]
        differences = []
        for j in range(len(opt_prims)):
            if j != i:
                imp1 = opt_prims[j][1]
                differences.append(imp1 - imp > epsilon)
        if all(differences):
            index = i

    if index is None:
        return None, None, None
    else:
        return opt_prims[index]
