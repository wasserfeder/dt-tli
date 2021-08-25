
import numpy as np
import sys
sys.path.append("/home/erfan/iitchs/catl_planning/python-stl/stl")
from stl import STLFormula, Operation, RelOperation



def make_stl_primitives1(signals):
    alw_gt = [STLFormula(Operation.ALWAYS, low=0, high=0, child=STLFormula(Operation.PRED, relation= RelOperation.GT, variable='x_{}'.format(i), threshold=0)) for i in range(len(signals[0]))]
    alw_le = [STLFormula(Operation.ALWAYS, low=0, high=0, child=STLFormula(Operation.PRED, relation= RelOperation.LE, variable='x_{}'.format(i), threshold=0)) for i in range(len(signals[0]))]
    return alw_gt + alw_le


def make_stl_primitives2(signals):
    alw_eve_gt = [STLFormula(Operation.ALWAYS, low=0, high=0, child=STLFormula(Operation.EVENT, low=0, high=0, child=STLFormula(Operation.PRED, relation= RelOperation.GT, variable='x_{}'.format(i), threshold=0))) for i in range(len(signals[0]))]
    alw_eve_le = [STLFormula(Operation.ALWAYS, low=0, high=0, child=STLFormula(Operation.EVENT, low=0, high=0, child=STLFormula(Operation.PRED, relation= RelOperation.LE, variable='x_{}'.format(i), threshold=0))) for i in range(len(signals[0]))]
    return alw_eve_gt + alw_eve_le



# def reverse_prim(primitive):
#
#


class STL_Param_Setter(object): # Done
    def __init__(self):
        self.counter = 0

    def set_pars(self, primitive, params):
        if primitive.op == 6 or primitive.op == 7:
            primitive.low = params[self.counter]
            primitive.high = params[self.counter+1]
            self.counter = self.counter + 2
            primitive.child = self.set_pars(primitive.child, params)
            return primitive
        elif primitive.op == 3:
            for i in range(len(primitive.children)):
                primitive.children[i] = self.set_pars(primitive.children[i],
                                        params)
                self.counter = self.counter + 1
            return primitive

        elif primitive.op == 8:
            primitive.threshold = params[self.counter]
            return primitive

        elif primitive.op == 5:
            primitive.low = params[self.counter]
            primitive.high = params[self.counter+1]
            self.counter = self.counter +  2
            primitive.left = self.set_pars(primitive.left, params)
            primitive.right = self.set_pars(primitive.right, params)
            return primitive
