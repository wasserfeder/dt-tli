
import stl_syntax
from stl_syntax import Signal, Formula, LE, GT, ALWAYS, EVENTUALLY, EXPR
import itertools
import numpy as np

import sys
sys.path.append("/home/erfan/Documents/University/Projects/Learning_Specifications/python-stl/stl")
from stl import STLFormula, Operation, RelOperation


class STLSignal(Signal):
    def __init__(self, index=0, op=LE, pi=0):
        self.index = index
        self.op = op
        self.pi = pi
        self.labels = [lambda t: [self.index, t]]
        self.f = lambda vs: (vs[0] - self.pi) * (-1 if self.op == LE else 1)

    def __deepcopy__(self, memo):
        return STLSignal(self.index, self.op, self.pi)

    def __str__(self):
        return "x_%d %s %.2f" % (self.index, "<=" if self.op == LE else ">", self.pi)


class STLFormula1(Formula):

    def __init__(self, live, index, op):
        if live:
            Formula.__init__(self, ALWAYS, [Formula(EXPR, [STLSignal(index, op)])], type=1)
        else:
            Formula.__init__(self, EVENTUALLY, [Formula(EXPR, [STLSignal(index, op)])], type=1)

    @property
    def index(self):
        return self.args[0].args[0].index

    @property
    def rel(self):
        return self.args[0].args[0].op

    @property
    def pi(self):
        return self.args[0].args[0].pi

    @pi.setter
    def pi(self, value):
        self.args[0].args[0].pi = value

    @property
    def t0(self):
        return self.bounds[0]

    @t0.setter
    def t0(self, value):
        self.bounds[0] = value

    @property
    def t1(self):
        return self.bounds[1]

    @t1.setter
    def t1(self, value):
        self.bounds[1] = value

    def reverse_rel(self):
        rel = self.args[0].args[0].op
        self.args[0].args[0].op = LE if rel == GT else GT

    def reverse_op(self):
        op = self.op
        self.op = 5 if op == 6 else 6





class STLFormula2(Formula):

    def __init__(self, live, index, op):
        if live:
            Formula.__init__(self, ALWAYS, [Formula(EVENTUALLY, [Formula(EXPR, [STLSignal(index, op)])])], type=2)
        else:
            Formula.__init__(self, EVENTUALLY, [Formula(ALWAYS, [Formula(EXPR, [STLSignal(index, op)])])], type=2)

    @property
    def index(self):
        return self.args[0].args[0].args[0].index

    @property
    def rel(self):
        return self.args[0].args[0].args[0].op

    @property
    def pi(self):
        return self.args[0].args[0].args[0].pi

    @pi.setter
    def pi(self, value):
        self.args[0].args[0].args[0].pi = value

    @property
    def t0(self):
        return self.bounds[0]

    @t0.setter
    def t0(self, value):
        self.bounds[0] = value

    @property
    def t1(self):
        return self.bounds[1]

    @t1.setter
    def t1(self, value):
        self.bounds[1] = value

    @property
    def t3(self):
        return self.args[0].bounds[1]

    @t3.setter
    def t3(self, value):
        self.args[0].bounds[1] = value

    def reverse_rel(self):
        rel = self.args[0].args[0].args[0].op
        self.args[0].args[0].args[0].op = LE if rel == GT else GT

    def reverse_op(self):
        outer_op = self.op
        if outer_op == 5:
            self.op = 6
            self.args[0].op = 5
        else:
            self.op = 5
            self.args[0].op = 6


# def set_stl1_pars(primitive, params):
#     primitive.pi = params[0]
#     primitive.t0 = int(params[1])
#     primitive.t1 = int(params[2])
#     return primitive
#
#
# def set_stl2_pars(primitive, params):
#     primitive.pi = params[0]
#     primitive.t0 = int(params[1])
#     primitive.t1 = int(params[2])
#     primitive.t3 = int(params[3])
#     return primitive



# def make_stl_primitives1(signals):
#     alw_gt = [STLFormula1(True, index, op) for index, op in itertools.product(range(len(signals[0])), [GT])]
#     alw_le = [STLFormula1(True, index, op) for index, op in itertools.product(range(len(signals[0])), [LE])]
#     return alw_gt + alw_le
#
#
#
# def make_stl_primitives2(signals):
#     alw_eve_gt = [STLFormula2(True, index, op) for index, op in itertools.product(range(len(signals[0])), [GT])]
#     alw_eve_le = [STLFormula2(True, index, op) for index, op in itertools.product(range(len(signals[0])), [LE])]
#     return alw_eve_gt + alw_eve_le


def set_stl1_pars(primitive, params):
    primitive.child.threshold   = params[0]
    primitive.low               = int(params[1])
    primitive.high              = int(params[2])
    return primitive


def set_stl2_pars(primitive, params):
    primitive.child.child.threshold = params[0]
    primitive.low                   = int(params[1])
    primitive.high                  = int(params[2])
    primitive.child.high            = int(params[3])
    return primitive


def reverse_primitive(primitive, primitive_type):
    if primitive_type == 1:
        if primitive.op == 6:
            primitive.op = 7
        else:
            primitive.op = 6
        if primitive.child.relation == 3:
            primitive.child.relation = 2
        else:
            primitive.child.relation = 3
    return primitive



def make_stl_primitives1(signals):
    alw_gt = [STLFormula(Operation.ALWAYS, low=0, high=0, child=STLFormula(Operation.PRED, relation= RelOperation.GT, variable='x_{}'.format(i), threshold=0)) for i in range(len(signals[0]))]
    alw_le = [STLFormula(Operation.ALWAYS, low=0, high=0, child=STLFormula(Operation.PRED, relation= RelOperation.LE, variable='x_{}'.format(i), threshold=0)) for i in range(len(signals[0]))]
    return alw_gt + alw_le


def make_stl_primitives2(signals):
    alw_eve_gt = [STLFormula(Operation.ALWAYS, low=0, high=0, child=STLFormula(Operation.EVENT, low=0, high=0, child=STLFormula(Operation.PRED, relation= RelOperation.GT, variable='x_{}'.format(i), threshold=0))) for i in range(len(signals[0]))]
    alw_eve_le = [STLFormula(Operation.ALWAYS, low=0, high=0, child=STLFormula(Operation.EVENT, low=0, high=0, child=STLFormula(Operation.PRED, relation= RelOperation.LE, variable='x_{}'.format(i), threshold=0))) for i in range(len(signals[0]))]
    return alw_eve_gt + alw_eve_le


def set_combined_stl_pars(primitive, primitive_type, params):
    if primitive_type == 3 or primitive_type == 4:
        primitive.low = int(params[-2])
        primitive.high = int(params[-1])
        children = primitive.child.children
        for j in range(len(children)):
            primitive.child.children[j].threshold = params[j]
        return primitive
