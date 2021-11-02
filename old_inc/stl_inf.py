
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

from stl_syntax import Formula, AND, OR, NOT, satisfies, robustness, GT
import numpy as np
from pso_test import run_pso_optimization
from pso import compute_robustness
from stl_prim import set_stl1_pars, set_stl2_pars

# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

class DTree(object):        # Decission tree recursive structure
    def __init__(self, primitive, signals, parent, left=None, right=None):
        self.primitive = primitive
        self.signals = signals
        self.parent = parent
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
        if self.right is not None and self.left is not None:
            return Formula(OR, [left, Formula(AND, [right, self.right.get_formula()])])
        else:
            return left


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================
def build_tree(signals, labels, rho_path, depth, primitives, D_t, args, parent):

    # Check stopping conditions
    if (depth <= 0) or (len(signals) == 0):
        return None

    prim, impurity, rhos = best_prim_pso(signals, labels, rho_path, primitives,
                                                                    D_t, args)
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
    tree = DTree(prim, signals, parent)
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
    tree.left = build_tree(sat_signals, sat_labels, sat_rho, depth - 1,
                primitives, sat_weights, args, tree)
    tree.right = build_tree(unsat_signals, unsat_labels, unsat_rho, depth - 1,
                primitives, unsat_weights, args, tree)

    return tree


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def best_prim_pso(signals, labels, rho_path, primitives, D_t, args):
    opt_prims = []
    for primitive in primitives:
        primitive = primitive.copy()
        print('***************************************************************')
        print("candidate primitive:", primitive)
        params, impurity = run_pso_optimization(signals, labels, rho_path,
                           primitive, D_t, args)
        if primitive.type == 1:
            primitive = set_stl1_pars(primitive, params)

        else:
            primitive = set_stl2_pars(primitive, params)

        rhos = np.array([compute_robustness(signals[i], params, primitive,
                                    rho_path[i]) for i in range(len(signals))])
        opt_prims.append([primitive, impurity, rhos])

    return min(opt_prims, key=lambda x: x[1])
