import numpy as np


class Pruning(object):
    def __init__(self, tree, depth):
        self.tree = tree
        self.depth = depth
        self.leaves = []
        find_leaves(self.tree)

    def find_leaves(self, tree):
        if tree.left is None and tree.right is None:
            self.leaves.append(tree)
        if tree.left is not None:
            find_leaves(tree.left)
        if tree.right is not None:
            find_leaves(tree.right)
