"""
Main inference module.

Contains the decision tree construction and related definitions.

Author: Erfan Aasi (eaasi@bu.edu)

"""
from stl_syntax import Formula, AND, OR, NOT, satisfies, robustness
# from stl_impurity import Optimize_Misclass_Gain
# from stl_linear_optimization import Optimize_Misclass_Gain
from stl_linear_optimization import Optimize_Misclass_Gain
from stl_prim import make_stl_primitives1, make_stl_primitives2, split_groups, SimpleModel
import numpy as np

class Traces(object):
    """
    Class to store a set of labeled signals
    """

    def __init__(self, signals=None, labels=None):
        """
        signals : list of m by n matrices
                  Last row should be the sampling times
        labels : list of labels
                 Each label should be either 1 or -1
        """
        self._signals = [] if signals is None else np.array(signals, dtype=float)
        self._labels = [] if labels is None else labels
        self.m, self.length = len(self._labels), len(self._signals[0][0])
        self._pos_indices, self._neg_indices = [], []

    @property
    def labels(self):
        return self._labels

    @property
    def signals(self):
        return self._signals

    def pos_indices(self):
        for i in range(len(self._labels)):
            if self._labels[i] >= 0:
                self._pos_indices.append(i)
        return self._pos_indices

    def neg_indices(self):
        for i in range(len(self._labels)):
            if self._labels[i] <= 0:
                self._neg_indices.append(i)
        return self._neg_indices


    def get_sindex(self, i):
        """
        Obtains the ith component of each signal

        i : integer
        """
        return self.signals[:, i]

    def as_list(self):
        """
        Returns the constructor arguments
        """
        return [self.signals, self.labels]

    def zipped(self):
        """
        Returns the constructor arguments zipped
        """
        return zip(*self.as_list())

class DTree(object):
    """
    Decission tree recursive structure

    """

    def __init__(self, primitive, traces, robustness=None,
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
        self._traces = traces
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



def learn_stlinf(signals, pdist, depth, inc, rho, stop_condition, disp):
    args = locals().copy()
    # Stopping condition
    if any([stop(args) for stop in stop_condition]):
        return None

    # Find primitive using impurity measure
    primitives1 = make_stl_primitives1(signals)
    primitives2 = make_stl_primitives2(signals)
    primitive, impurity = _find_best_primitive(signals, primitives1, primitives2, rho, pdist, disp)
    if disp:
        print primitive

    # Classify using best primitive and split into groups
    tree = DTree(primitive, traces)
    prim_rho = [robustness(primitive, SimpleModel(s)) for s in traces.signals]
    if rho is None:
        rho = [np.inf for i in traces.labels]
    # [prim_rho, rho, signals, label]
    sat_, unsat_ = split_groups(zip(prim_rho, rho, *traces.as_list()),
        lambda x: x[0] >= 0)

    # Switch sat and unsat if labels are wrong. No need to negate prim rho since
    # we use it in absolute value later
    if len([t for t in sat_ if t[3] >= 0]) < \
        len([t for t in unsat_ if t[3] >= 0]):
        sat_, unsat_ = unsat_, sat_
        tree.primitive = Formula(NOT, [tree.primitive])

    # No further classification possible
    if len(sat_) == 0 or len(unsat_) == 0:
        return None

    # Redo data structures
    sat, unsat = [(Traces(*group[2:]),
                   np.amin([np.abs(group[0]), group[1]], 0))
                   for group in [zip(*sat_), zip(*unsat_)]]

    # Recursively build the tree
    tree.left = bdt_stlinf_(sat[0], sat[1], depth - 1, pdist, inc,
                        optimize_impurity, stop_condition)
    tree.right = bdt_stlinf_(unsat[0], unsat[1], depth - 1, pdist, inc,
                         optimize_impurity, stop_condition)

    return tree

def perfect_stop(kwargs):
    """
    Returns True if all traces are equally labeled.
    """
    return all([l > 0 for l in kwargs['traces'].labels]) or \
        all([l <= 0 for l in kwargs['traces'].labels])

def depth_stop(kwargs):
    """
    Returns True if the maximum depth has been reached
    """
    return kwargs['depth'] <= 0

def _find_best_primitive(signals, primitives1, primitives2, robustness, pdist, disp):
    # Parameters will be set for the copy of the primitive
    opt_prims = []
    for primitive in primitives1:
        optimize_impurity = Optimize_Misclass_Gain(signals. primitive.copy(), 1, robustness, pdist, disp)
        opt_prims.append(optimize_impurity.get_solution())

    for primitive in primitives2:
        optimize_impurity = Optimize_Misclass_Gain(signals. primitive.copy(), 2, robustness, pdist, disp)
        opt_prims.append(optimize_impurity.get_solution())

    return min(opt_prims, key=lambda x: x[1])
