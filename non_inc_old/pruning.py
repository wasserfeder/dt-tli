
import numpy as np





class Pruning_Class(object):
    def __init__(self, signals, labels, rho_path, D_t):
        self.signals = signals
        self.labels = labels
        self.rho_path = rho_path
        self.D_t = D_t


    def check_combination(self, root_prim, child_prim, direction):
        if direction == 'right':
            root_prim.reverse_rel()
            root_prim.reverse_op()

        if (root_prim.op == 5) and (child_prim.op == 5):    # Always and always
            alw_alw = Prune_STL(5, 5, pred_1, pred_2)
