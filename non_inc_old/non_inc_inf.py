# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

# from stl_syntax import Formula, AND, OR, NOT, satisfies, robustness, GT
import numpy as np
from pso_test import run_pso_optimization
from pso import compute_robustness
from stl_prim import STL_Param_Setter

import copy


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
        if self.right is not None and self.left is not None:
            return Formula(OR, [left, Formula(AND, [right, self.right.get_formula()])])
        else:
            return left


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================
class Learning_Class(object):
    def __init__(self, primitives, args):
        self.args       = args
        self.primitives = primitives
        self.prunes = []
        self.stl_param_setter = STL_Param_Setter()


    def normal_tree(self, data, rho_path, depth, D_t):
        signals, traces, labels = data["tr_s"], data["tr_t"], data["tr_l"]
        if (depth <= 0) or (len(signals) == 0):
            return None

        info = self.best_prim_pso(data, rho_path, D_t)
        prim, impurity, rhos = info['primitive'], info['impurity'], info['rhos']
        print('***************************************************************')
        print('Primitive:', prim)
        print('impurity:', impurity)
        tree = DTree(prim, signals)

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
        sat_data = {"tr_s": sat_signals, "tr_t": sat_traces, "tr_l": sat_labels}
        unsat_data = {"tr_s": unsat_signals, "tr_t": unsat_traces, "tr_l": unsat_labels}

        # Recursively build the tree
        tree.left = self.normal_tree(sat_data, sat_rho, depth - 1,
                    sat_weights)
        tree.right = self.normal_tree(unsat_data, unsat_rho, depth - 1,
                    unsat_weights)

        return tree



    def pruned_tree(self, signals, labels, rho_path, prim, impurity, rhos, depth, D_t): #include the pruning_record
        if (depth <= 0) or (len(signals) == 0):
            return None

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

        left_prim, left_impurity, left_rhos = self.best_prim_pso(sat_signals, sat_labels, sat_rho, sat_weights)
        combined_prim, combined_impurity, combined_rhos = self.combine_primitives(prim, left_prim, signals, labels, rho_path, D_t, 'left')            # write the "combine_primitives" method

        if combined_impurity >= impurity:
            self.prunes.append((depth, prim, left_prim, combined_prim))
            tree = self.build_pruned_tree(signals, labels, rho_path, combined_prim, combined_impurity, combined_rhos, depth, D_t)

        right_prim, right_impurity, right_rhos = self.best_prim_pso(unsat_signals, unsat_labels, unsat_rho, unsat_weights)
        combined_prim, combined_impurity, combined_rhos = self.combine_primitives(prim, right_prim, signals, rho_path, D_t, 'right')

        if combined_impurity >= impurity:
            self.prunes.append((depth, prim, right_prim, combined_prim))
            tree = self.build_pruned_tree(signals, labels, rho_path, combined_prim, combined_impurity, combined_rhos, depth, D_t)

        tree = DTree(prim, signals)
        tree.left = self.build_pruned_tree(sat_signals, sat_labels, sat_rho, left_prim, left_impurity, left_rhos, depth-1, sat_weights)
        tree.right = self.build_pruned_tree(unsat_signals, unsat_labels, unsat_rho, right_prim, right_impurity, right_rhos, depth-1, unsat_weights)

        return tree, self.prunes



    def best_prim(self, data, rho_path, D_t):
        signals, traces, labels = data["tr_s"], data["tr_t"], data["tr_l"]
        opt_prims = []
        for primitive in self.primitives:
            primitive = copy.deepcopy(primitive)
            print('******************************************************')
            print("candidate primitive:", primitive)
            params, impurity = run_pso_optimization(signals, traces, labels,
                            rho_path, primitive, D_t, self.args)
            self.stl_param_setter.counter = 0
            primitive = self.stl_param_setter.set_pars(primitive, params)

            rhos = np.array([compute_robustness(traces[i], primitive, rho_path[i]) for i in range(len(signals))])
            opt_prims.append([primitive, impurity, rhos])

        prim, impurity, rhos =  min(opt_prims, key=lambda x: x[1])
        # Check for EVENTUALLY primitives
        # counter = 0
        # reverse_counter = 0
        # for i in range(len(signals)):
        #     if (rhos[i] >= 0 and labels[i] < 0) or (rhos[i] < 0 and labels[i] > 0):
        #         counter = counter + 1
        #     if (-rhos[i] >= 0 and labels[i] < 0) or (-rhos[i] < 0 and labels[i]> 0):
        #         reverse_counter = reverse_counter + 1
        #
        # if reverse_counter < counter:
        #     prim.reverse_rel()
        #     prim.reverse_op()
        #     rhos = - rhos
        info = {'primitive': prim, 'impurity': impurity, 'rhos': rhos}
        return info



    def combine_primitives(self, root_prim, child_prim, signals, labels, rho_path, D_t, direction):
        pruning_class = Pruning_Class(signals, labels, rho_path, D_t)
        prim, impurity, rhos = pruning_class.check_combination(root_prim, child_prim, direction)
        return prim, impurity, rhos
