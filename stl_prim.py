"""
Module with depth 2 p-stl definitions

Author: Francisco Penedo (franp@bu.edu)

"""
import stl_syntax
from stl_syntax import Signal, Formula, LE, GT, ALWAYS, EVENTUALLY, EXPR
import itertools
# from bisect import bisect_left
import numpy as np
from pyparsing import Word, alphas, Suppress, Optional, Combine, nums, \
    Literal, alphanums, Keyword, Group, ParseFatalException, MatchFirst


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
            Formula.__init__(self, ALWAYS, [Formula(EXPR, [STLSignal(index, op)])])
        else:
            Formula.__init__(self, ALWAYS, [Formula(EXPR, [STLSignal(index, op)])])

    @property
    def index(self):
        return self.args[0].args[0].index

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

    def reverse_op(self):
        """
        Reverses the operator of the predicate
        """
        op = self.args[0].args[0].op
        self.args[0].args[0].op = LE if op == GT else GT



class STLFormula2(Formula):

    def __init__(self, live, index, op):
        if live:
            Formula.__init__(self, ALWAYS, [Formula(EVENTUALLY, [Formula(EXPR, [STLSignal(index, op)])])])
        else:
            Formula.__init__(self, EVENTUALLY, [Formula(ALWAYS, [Formula(EXPR, [STLSignal(index, op)])])])

    @property
    def index(self):
        return self.args[0].args[0].args[0].index

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

    def reverse_op(self):
        op = self.args[0].args[0].args[0].op
        self.args[0].args[0].args[0].op = LE if op == GT else GT


def set_stl1_pars(primitive, t0, t1, pi):
    primitive.t0 = t0
    primitive.t1 = t1
    primitive.pi = pi


def set_stl2_pars(primitive, t0, t1, t3, pi):
    primitive.t0 = t0
    primitive.t1 = t1
    primitive.t3 = t3
    primitive.pi = pi


class SimpleModel(object):
    def __init__(self, signals):
        self._signals = signals
        self._tinter = signals[-1][1] - signals[-1][0]
        self._lsignals = len(signals[-1])

    def getVarByName(self, indices):
        tindex = min(
            np.floor(indices[1]/self._tinter), self._lsignals - 1)
        return self._signals[indices[0]][tindex]

    @property
    def tinter(self):
        return self._tinter


def make_stl_primitives1(signals):
    alw_gt = [STLFormula1(True, index, op) for index, op in itertools.product(range(len(signals[0])), [GT])]
    alw_le = [STLFormula1(False, index, op) for index, op in itertools.product(range(len(signals[0])), [LE])]
    return alw_gt + alw_le



def make_stl_primitives2(signals):
    alw_ev = [
        STLFormula2(True, index, op)
        for index, op
        in itertools.product(range(len(signals.traces[0])), [LE])
    ]
    ev_alw = [
        STLFormula2(False, index, op)
        for index, op
        in itertools.product(range(len(signals.traces[0])), [LE])
    ]
    return alw_ev + ev_alw




def split_groups(l, group):
    p = [x for x in l if group(x)]
    n = [x for x in l if not group(x)]
    return p, n


# parser

def expr_parser():
    num = stl.num_parser()

    T_UND = Suppress(Literal("_"))
    T_LE = Literal("<=")
    T_GR = Literal(">")

    integer = Word(nums).setParseAction(lambda t: int(t[0]))
    relation = (T_LE | T_GR).setParseAction(lambda t: LE if t[0] == "<=" else GT)
    expr = Suppress(Word(alphas)) + T_UND + integer + relation + num
    expr.setParseAction(lambda t: STLSignal(t[0], t[1], t[2]))

    return expr

def llt_parser():
    stl_parser = MatchFirst(stl.stl_parser(expr_parser()))
    return stl_parser
