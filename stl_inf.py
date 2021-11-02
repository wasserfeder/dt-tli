
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np
from pso_test import run_pso_optimization
from pso import compute_robustness
from stl_prim import set_stl1_pars, set_stl2_pars, reverse_primitive
from stl import STLFormula, Operation
import copy

# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

class DTree(object):        # Decission tree recursive structure
    def __init__(self, primitive, left=None, right=None):
        self.primitive = primitive
        self.primitive_type = self.get_primitive_type(self.primitive)
        self.params = self.get_primitive_parameters(self.primitive)
        self.left = left
        self.right = right


    def classify(self, signal):
        if self.primitive_type == 1 or self.primitive_type == 2:
            rho = compute_robustness([signal], self.params, self.primitive, self.primitive_type, [np.inf])
        else:
            rho = compute_combined_robustness([signal], self.params, self.primitive, self.primitive_type, [np.inf])
        if rho[0] >= 0:
            if self.left is None:
                return 1
            else:
                return self.left.classify(signal)
        else:
            if self.right is None:
                return -1
            else:
                return self.right.classify(signal)


    def get_primitive_type(self, primitive):
        if primitive.op == 6 or primitive.op == 7:
            if primitive.child.op == 8:
                primitive_type = 1
            elif primitive.child.op == 6 or primitive.child.op == 7:
                primitive_type = 2
            elif primitive.child.op == 3:
                if primitive.op == 6:
                    primitive_type = 3
                else:
                    primitive_type = 4
            if primitive.child.op == 5:
                primitive_type = 5
            return primitive_type


    def get_primitive_parameters(self, primitive):
        t0, t1 = int(primitive.low), int(primitive.high)
        if self.primitive_type == 1:
            threshold = primitive.child.threshold
            params = [threshold, t0, t1]
        elif self.primitive_type == 2:
            threshold = primitive.child.child.threshold
            t3 = int(primitive.child.high)
            params = [threshold, t0, t1, t3]
        elif self.primitive_type == 3 or 4:
            children = primitive.child.children
            params = []
            for child in children:
                params += [child.threshold]
            params += [t0, t1]
        elif self.primitive_type == 5:
            left_threshold = self.primitive.child.left.threshold
            right_threshold = self.primitive.child.right.threshold
            t3 = self.primitive.child.high
            params = [left_threshold, right_threshold, t0, t1, t3]
        return params




    def tree_robustness(self, trace, rho_path):
        rho_primitive = self.primitive.robustness(trace, 0)
        rho = np.min([rho_primitive, rho_path])
        if self.left is None:
            return rho
        elif rho >= 0 and self.left is not None:
            return self.left.tree_robustness(trace, rho)
        else:
            return self.right.tree_robustness(trace, -rho)



    def get_formula(self):
        left = self.primitive
        right = STLFormula(Operation.NOT, child = self.primitive)
        if self.left is not None:
            left = STLFormula(Operation.AND, children = [self.primitive, self.left.get_formula()])
        if self.right is not None and self.left is not None:
            return STLFormula(Operation.OR, children = [left, STLFormula(Operation.AND, children = [right, self.right.get_formula()])])
        else:
            return left



# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================
def inc_tree(signals, traces, labels, pos_ind, neg_ind, rho_path, depth, primitives, D_t, args):
    # Check stopping conditions
    if (depth <= 0) or (len(signals) == 0):
        return None

    distance = compute_distance(signals, pos_ind, neg_ind)      # TODO: write the compute_distance function



    prim, impurity, rhos = best_prim(signals, traces, labels, rho_path, primitives, D_t, args)

    print('***************************************************************')
    print('Depth:', args.depth - depth + 1)
    print('Primitive:', prim)
    print('impurity:', impurity)
    tree = DTree(prim)
    sat_signals, unsat_signals  = [], []            # TODO: We have to update pos_ind and neg_ind after partitioning.It will be like "sat_pos_ind, sat_neg_ind, unsat_pos_ind, unsat_neg_ind"
    sat_traces, unsat_traces    = [], []
    sat_indices, unsat_indices  = [], []
    sat_rho, unsat_rho          = [], []
    sat_labels, unsat_labels    = [], []
    sat_weights, unsat_weights  = [], []

    for i in range(len(signals)):
        if rhos[i] >= 0:
            sat_signals.append(signals[i])
            sat_traces.append(traces[i])
            sat_rho.append(rhos[i])
            sat_labels.append(labels[i])
            sat_indices.append(i)
            sat_weights.append(D_t[i])

        else:
            unsat_signals.append(signals[i])
            unsat_traces.append(traces[i])
            unsat_rho.append(-rhos[i])
            unsat_labels.append(labels[i])
            unsat_indices.append(i)
            unsat_weights.append(D_t[i])

    sat_signals     = np.array(sat_signals)
    unsat_signals   = np.array(unsat_signals)

    # Recursively build the tree
    tree.left = inc_tree(sat_signals, sat_traces, sat_labels, sat_rho, depth - 1,
                primitives, sat_weights, args)
    tree.right = inc_tree(unsat_signals, unsat_traces, unsat_labels, unsat_rho, depth - 1,
                primitives, unsat_weights, args)

    return tree


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def best_prim(signals, traces, labels, rho_path, primitives, D_t, args):
    opt_prims = []
    for primitive in primitives:
        primitive = copy.deepcopy(primitive)
        print('***************************************************************')
        print("candidate primitive:", primitive)
        if primitive.child.op == 8:
            primitive_type = 1
        else:
            primitive_type = 2
        params, impurity = run_pso_optimization(signals, traces, labels, rho_path,
                           primitive, primitive_type, D_t, args)
        if primitive_type == 1:
            primitive = set_stl1_pars(primitive, params)

        else:
            primitive = set_stl2_pars(primitive, params)

        rhos = np.array(compute_robustness(signals, params, primitive, primitive_type, rho_path))
        opt_prims.append([primitive, impurity, rhos])

    prim, impurity, rhos =  min(opt_prims, key=lambda x: x[1])
    counter = 0
    reverse_counter = 0
    for i in range(len(signals)):
        if (rhos[i] >= 0 and labels[i] < 0) or (rhos[i] < 0 and labels[i] > 0):
            counter = counter + 1
        if (-rhos[i] >= 0 and labels[i] < 0) or (-rhos[i] < 0 and labels[i]> 0):
            reverse_counter = reverse_counter + 1

    if reverse_counter < counter:
        if prim.child.op == 8:
            primitive_type = 1
        else:
            primitive_type = 2
        prim = reverse_primitive(prim, primitive_type)
        rhos = - rhos

    return prim, impurity, rhos
