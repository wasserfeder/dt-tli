"""
Main inference module.

Contains the decision tree construction and related definitions.

Author: Erfan Aasi (eaasi@bu.edu)

"""
from stl_syntax import Formula, AND, OR, NOT, satisfies, robustness, GT
import numpy as np
from test_optimization_problem_sum_interval import PrimitiveMILP
import pickle


class DTree(object):
    """
    Decission tree recursive structure

    """

    def __init__(self, primitive, signals, robustness=None,
                 left=None, right=None):
        """
        primitive : a LLTFormula object
                    The node's primitive
        traces : a Traces object
                 The traces used to build this node
        robustness : a list of numeric. Not used
        left : a DTree object. Optional
               The subtree corresponding to an unsat result to this node's test
        right : a DTree object. Optional
                The subtree corresponding to a sat result to this node's test
        """
        self._primitive = primitive
        self.signals = signals
        self._robustness = robustness
        self._left = left
        self._right = right

    def classify(self, signal):
        """
        Classifies a signal. Returns a label 1 or -1

        signal : an m by n matrix
                 Last row should be the sampling times
        """
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
        """
        Obtains an STL formula equivalent to this tree
        """
        left = self.primitive
        right = Formula(NOT, [self.primitive])
        if self.left is not None:
            left = Formula(AND, [
                self.primitive,
                self.left.get_formula()
            ])
        if self.right is not None:
            return Formula(OR, [left,
                                Formula(AND, [
                                    right,
                                    self.right.get_formula()
            ])])
        else:
            return left

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        self._left = value

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        self._right = value

    @property
    def primitive(self):
        return self._primitive

    @primitive.setter
    def primitive(self, value):
        self._primitive = value

    @property
    def robustness(self):
        return self._robustness

    @robustness.setter
    def robustness(self, value):
        self._robustness = value



def build_tree(signals, labels, depth, primitives1, rho_path):
    # Check stopping conditions
    if depth <= 0:
        return None


    primitive, impurity, robustnesses = find_best_primitive(signals, labels, primitives1, rho_path)
    print('Primitive:', primitive)
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
    dict = {'sat_indices':sat_indices, 'unsat_indices':unsat_indices, 'sat_rho': sat_rho, 'unsat_rho':unsat_rho}
    pickle_out = open("indices.pickle", "wb")
    pickle.dump(dict, pickle_out)
    pickle_out.close()



    # Recursively build the tree
    tree.left = build_tree(sat_signals, sat_labels, depth - 1, primitives1, sat_rho)
    tree.right = build_tree(unsat_signals, unsat_labels, depth - 1, primitives1, unsat_rho)

    return tree





def find_best_primitive(signals, labels, primitives1, rho_path):
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
        primitive.t0 = milp.get_interval()[0]
        primitive.t1 = milp.get_interval()[1]
        rho = milp.get_robustnesses()
        opt_prims.append([primitive, milp.model.objVal, rho])

    return min(opt_prims, key=lambda x: x[1])
