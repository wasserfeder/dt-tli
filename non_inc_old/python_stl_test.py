

import sys
sys.path.append("/home/erfan/iitchs/catl_planning/python-stl/stl")
from stl import STLFormula, Operation, RelOperation, Trace
import copy
from stl_prim import STL_Param_Setter
from pso_test import get_indices



i = 1
j = 2
pred_1 = STLFormula(Operation.PRED, relation= RelOperation.GT, variable='x_{}'.format(i), threshold=10)
pred_2 = STLFormula(Operation.PRED, relation= RelOperation.GT, variable='x_{}'.format(j), threshold=20)
pred = STLFormula(Operation.AND, children = [pred_1, pred_2])
alw = STLFormula(Operation.ALWAYS, low=6, high=9, child=pred)
unt = STLFormula(Operation.UNTIL, low=5, high=7, left = pred, right=pred)
eve_alw = STLFormula(Operation.EVENT, low=1, high=2, child=alw)
# print(copy.deepcopy(alw))
# print(alw.op)
# print(alw.child.children[0].op)
# print(alw.child.children[0].relation)
# print(alw.child.children[0].variable.split("_")[1])
# print(unt.__str__())
# print(unt.op)
#
#
# eve = STLFormula(Operation.EVENT, low=1, high=3, child=pred_1)
# varnames = ['x_1', 'x_2']
# data = [[3.2,4.7,11.65,9.1,8], [2,7,9,4,8]]
# timepoints = [0, 1, 2, 3, 4]
# s = Trace(varnames, timepoints, data)
#
# print(eve.robustness(s, 0))

# stl_param_setter = STL_Param_Setter()
# stl_param_setter.counter = 0
# params = [5, 9, 8, 12, 4, 6]
# eve_alw = stl_param_setter.set_pars(eve_alw, params)
# print(eve_alw)


indices = get_indices(alw, [])
print(indices)
