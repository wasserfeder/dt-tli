
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np
from pso_test import run_pso_optimization
from pso import compute_robustness
from stl_prim import set_stl1_pars, reverse_primitive
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
        self.children = []

    def classify(self, signal):
        rho = compute_robustness([signal], self.params, self.primitive, self.primitive_type, [np.inf])
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


    def tree_robustness(self, signals, rho_path):
        rhos = compute_robustness(signals, self.params, self.primitive, self.primitive_type, rho_path)
        for i in range(len(rhos)):
            if rhos[i] >= 0 and self.left is not None:
                rhos[i] = self.left.tree_robustness([signals[i]], [rhos[i]])
            elif rhos[i] < 0 and self.right is not None:
                rhos[i] = self.right.tree_robustness([signals[i]], [rhos[i]])
        return rhos




    def get_primitive_type(self, primitive):
        if primitive.op == 6 or primitive.op == 7:
            if primitive.child.op == 8:
                primitive_type = 1
            else:
                primitive_type = 2
            return primitive_type


    def get_primitive_parameters(self, primitive):
        t0, t1 = int(primitive.low), int(primitive.high)
        if self.primitive_type == 1:
            threshold = primitive.child.threshold
            params = [threshold, t0, t1]
        else:
            threshold = primitive.child.child.threshold
            t3 = int(primitive.child.high)
            params = [threshold, t0, t1, t3]
        return params


    # def tree_robustness(self, trace, rho_path):
    #     rho_primitive = self.primitive.robustness(trace, 0)
    #     rho = np.min([rho_primitive, rho_path])
    #     if self.left is None:
    #         return rho
    #     elif rho >= 0 and self.left is not None:
    #         return self.left.tree_robustness(trace, rho)
    #     else:
    #         return self.right.tree_robustness(trace, -rho)



    def get_formula(self):
        formula = self.primitive
        if self.left is None and self.right is None:
            return formula
        if self.left is not None and self.right is None:
            formula = STLFormula(Operation.AND, children = [formula, self.left.get_formula()])
            return formula
        if self.left is None and self.right is not None:
            right = STLFormula(Operation.NOT, child = self.primitive)
            formula = STLFormula(Operation.AND, children = [right, self.right.get_formula()])
            return formula
        else:
            left = STLFormula(Operation.AND, children = [self.primitive, self.left.get_formula()])
            right = STLFormula(Operation.NOT, child = self.primitive)
            return STLFormula(Operation.OR, children = [left, STLFormula(Operation.AND, children = [right, self.right.get_formula()])])


    def get_children(self):
        self.children.append(self.primitive)
        if self.left is not None:
            self.children.extend(self.left.get_children())
        if self.right is not None:
            self.children.extend(self.right.get_children())
        return self.children


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================
def inc_tree(signals, labels, rho_path, depth, primitives, args):
    if (depth <= 0) or (len(signals) == 0):
        return None

    prim, impurity, rhos = best_prim(signals, labels, rho_path, primitives,args)
    print('***************************************************************')
    print('Depth:', args.depth - depth + 1)
    print('Primitive:', prim)
    print('impurity:', impurity)

    tree = DTree(prim)
    pos_rho_ind, neg_rho_ind    = np.where(rhos >= 0)[0], np.where(rhos < 0)[0]
    sat_rho, unsat_rho          = rhos[pos_rho_ind], rhos[neg_rho_ind]
    sat_signals, sat_labels     = signals[pos_rho_ind], labels[pos_rho_ind]
    unsat_signals, unsat_labels = signals[neg_rho_ind], labels[neg_rho_ind]

    tree.left = inc_tree(sat_signals, sat_labels, sat_rho, depth-1,
                primitives, args)
    tree.right = inc_tree(unsat_signals, unsat_labels, unsat_rho, depth-1,
                primitives, args)

    return tree


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def best_prim(signals, labels, rho_path, primitives, args):
    opt_prims = []
    for primitive in primitives:
        primitive = copy.deepcopy(primitive)
        print('***************************************************************')
        print("candidate primitive:", primitive)
        if primitive.child.op == 8:
            primitive_type = 1
        else:
            primitive_type = 2
        params, impurity = run_pso_optimization(signals, labels, rho_path,
                           primitive, primitive_type, args)
        if primitive_type == 1:
            primitive = set_stl1_pars(primitive, params)

        rhos = np.array(compute_robustness(signals, params, primitive,
                                            primitive_type, rho_path))
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
