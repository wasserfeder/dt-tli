
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

from stl_syntax import Formula, AND, OR, NOT, satisfies, robustness, GT
import numpy as np
from pso_test import run_pso_optimization
from pso import compute_robustness
from stl_prim import set_stl1_pars, set_stl2_pars, reverse_primitive, set_combined_stl_pars
from stl import STLFormula, Operation
import copy
from combined_pso_test import run_combined_pso
from combined_pso import compute_combined_robustness

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

    # def classify(self, trace):
    #     rho = self.primitive.robustness(trace, 0)
    #     if rho >= 0:
    #         if self.left is None:
    #             return 1
    #         else:
    #             return self.left.classify(trace)
    #     else:
    #         if self.right is None:
    #             return -1
    #         else:
    #             return self.right.classify(trace)


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
def pruned_tree(signals, traces, labels, rho_path, depth, prim, impurity, rhos, primitives, D_t, args, prunes):
    if (depth <= 0) or (len(signals) == 0):
            return None, prunes

    pos_counter, neg_counter = 0, 0
    for i in range(len(signals)):
        if labels[i] == 1:
            pos_counter += 1
        else:
            neg_counter += 1
    if (pos_counter/len(signals) >= 0.99) or (neg_counter/len(signals) >= 0.99):
        return None, prunes

    tree = DTree(prim)
    print('***************************************************************')
    print("Depth:", args.depth - depth + 1)
    print('Root Primitive:', prim)
    print('Root impurity:', impurity)

    sat_signals, unsat_signals  = [], []
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

    if (len(sat_signals) != 0) and (depth - 1 > 0):
        left_prim, left_impurity, left_rhos = best_prim(sat_signals, sat_traces, sat_labels, sat_rho, primitives, sat_weights, args)
        print('***************************************************************')
        print('Left Primitive:', left_prim)
        combined_prim, combined_impurity, combined_rhos = combine_primitives(prim, left_prim, signals, traces, labels, rho_path, D_t, args, 'left')            # write the "combine_primitives" method

        if (combined_impurity is not None) and (combined_impurity <= impurity):
            prunes.append((args.depth - depth + 1, prim, left_prim, combined_prim))
            return pruned_tree(signals, traces, labels, rho_path, depth, combined_prim, combined_impurity, combined_rhos, primitives, D_t, args, prunes)

    if (len(unsat_signals) != 0) and (depth -1 > 0):
        right_prim, right_impurity, right_rhos = best_prim(unsat_signals, unsat_traces, unsat_labels, unsat_rho, primitives, unsat_weights, args)
        print('***************************************************************')
        print('Right Primitive:', right_prim)
        combined_prim, combined_impurity, combined_rhos = combine_primitives(prim, right_prim, signals, traces, labels, rho_path, D_t, args, 'right')

        if (combined_impurity is not None) and (combined_impurity <= impurity):
            prunes.append((args.depth - depth + 1, prim, right_prim, combined_prim))
            return pruned_tree(signals, traces, labels, rho_path, depth, combined_prim, combined_impurity, combined_rhos, primitives, D_t, args, prunes)


    if (len(sat_signals) != 0) and (depth - 1 > 0):
        tree.left, prunes = pruned_tree(sat_signals, sat_traces, sat_labels, sat_rho, depth-1, left_prim, left_impurity, left_rhos, primitives, sat_weights, args, prunes)
    if (len(unsat_signals) != 0) and (depth -1 > 0):
        tree.right, prunes = pruned_tree(unsat_signals, unsat_traces, unsat_labels, unsat_rho, depth-1, right_prim, right_impurity, right_rhos, primitives, unsat_weights, args, prunes)

    return tree, prunes



# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================
def normal_tree(signals, traces, labels, rho_path, depth, primitives, D_t, args):
    # Check stopping conditions
    if (depth <= 0) or (len(signals) == 0):
        return None

    prim, impurity, rhos = best_prim(signals, traces, labels, rho_path, primitives, D_t, args)

    print('***************************************************************')
    print('Depth:', args.depth - depth + 1)
    print('Primitive:', prim)
    print('impurity:', impurity)
    tree = DTree(prim)
    sat_signals, unsat_signals  = [], []
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
    tree.left = normal_tree(sat_signals, sat_traces, sat_labels, sat_rho, depth - 1,
                primitives, sat_weights, args)
    tree.right = normal_tree(unsat_signals, unsat_traces, unsat_labels, unsat_rho, depth - 1,
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

        # rhos = np.array([compute_robustness(traces[i], params, primitive, primitive_type,
        #                             rho_path[i]) for i in range(len(signals))])
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



# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def combine_primitives(root_prim, child_prim, signals, traces, labels, rho_path, D_t, args, direction):
    if (root_prim.op == 6 and (root_prim.child.op == 3 or root_prim.child.op == 8) and child_prim.op == 6) or (root_prim.op == 7 and child_prim.op == 7):
        if root_prim.child.op == 3:
            children = copy.deepcopy(root_prim.child.children)
            for child in children:
                child.threshold = 0
        else:
            children = [copy.deepcopy(root_prim.child)]
            children[0].threshold = 0
        children += [copy.deepcopy(child_prim.child)]
        children[-1].threshold = 0
        for i in range(len(children)-1):
            if (children[i].variable == children[-1].variable) and (children[i].relation == children[-1].relation):
                return None, None, None
        combined_pred = STLFormula(Operation.AND, children = children)
        if root_prim.op == 6:
            combined_prim = STLFormula(Operation.EVENT, low = 0, high = 0, child = combined_pred)
        else:
            combined_prim = STLFormula(Operation.ALWAYS, low = 0, high = 0, child = combined_pred)
        print('***************************************************************')
        print("candidate combined primitive:", combined_prim)
        prim, impurity, rhos = best_combined_prim(signals, traces, labels, rho_path, combined_prim, D_t, args)
        return prim, impurity, rhos


    # elif (root_prim.op == 6 and root_prim.child.op == 8 and child_prim.op == 7) or (root_prim.op == 7 and root_prim.child.op == 8 and child_prim.op == 6):
    #     if root_prim.op == 6:
    #         left_child = copy.deepcopy(child_prim.child)
    #         right_child = copy.deepcopy(root_prim.child)
    #     else:
    #         left_child = copy.deepcopy(root_prim.child)
    #         right_child = copy.deepcopy(child_prim.child)
    #     left_child.threshold = 0
    #     right_child.threshold = 0
    #     combined_pred = STLFormula(Operation.UNTIL, low = 0, high = 0, left = left_child, right = right_child)
    #     combined_prim = STLFormula(Operation.EVENT, low = 0, high = 0, child = combined_pred)
    #     print('***************************************************************')
    #     print("candidate combined primitive:", combined_prim)
    #     prim, impurity, rhos = best_combined_prim(signals, traces, labels, rho_path, combined_prim, D_t, args)
    #     return prim, impurity, rhos

    else:
        return None, None, None




# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def best_combined_prim(signals, traces, labels, rho_path, combined_prim, D_t, args):
    combined_prim = copy.deepcopy(combined_prim)
    if combined_prim.op == 6 and combined_prim.child.op == 3:
        primitive_type = 3
    elif combined_prim.op == 7 and combined_prim.child.op == 3:
        primitive_type = 4
    elif combined_prim.op == 6 and combined_prim.child.op == 5:
        primitive_type = 5
    params, impurity = run_combined_pso(signals, traces, labels, rho_path, combined_prim, primitive_type, D_t, args)

    prim = set_combined_stl_pars(combined_prim, primitive_type, params)
    rhos = compute_combined_robustness(signals, params, prim, primitive_type, rho_path)
    # rhos = [np.min([prim.robustness(traces[i],0), rho_path[i]]) for i in range(len(signals))]
    print('***************************************************************')
    print("best combined primitive:", prim)
    print("combined primitive impurity:", impurity)
    return prim, impurity, rhos
