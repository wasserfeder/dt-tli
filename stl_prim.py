
# import stl_syntax
import itertools
import numpy as np

import sys
sys.path.append("C:\TERRAA\ManeuverGame\python_stl")
from stl import STLFormula, Operation, RelOperation


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
    elif primitive_type == 5:
        primitive.low = int(params[2])
        primitive.high = int(params[3])
        primitive.child.high = int(params[4])
        primitive.child.left.threshold = params[0]
        primitive.child.right.threshold = params[1]
        return primitive
