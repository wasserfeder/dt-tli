import torch
import numpy as np
import math

import sys
import os
import glob

try:
    sys.path.append(glob.glob('python-stl/stl')[0])
except IndexError:
    pass

from antlr4 import InputStream, CommonTokenStream
# sys.path.append('..')
from stl import Operation, RelOperation, STLFormula
from stlLexer import stlLexer
from stlParser import stlParser
from stl import STLAbstractSyntaxTreeExtractor
from stl import Trace

def forward_disjunction(X, w):
    # data normalization
    alpha = 1.0
    r = -X
    s = torch.clone(r)
    s.apply_(lambda x: math.exp(-x/alpha))
    w_norm = w / w.sum()
    s_norm = s/s.sum()
    denominator = torch.mul(s_norm, w_norm)
    denominator = denominator.sum()

    # disjunction numberator


    numerator = torch.mul(s_norm, w_norm)
    numerator = torch.mul(numerator, r)
    numerator = numerator.sum()
    robust = -numerator/denominator
    return robust

def forward_conjunction(X, w):
    # data normalization
    alpha = 1.0
    s = torch.clone(X)
    s.apply_(lambda x: math.exp(-x/alpha))
    w_norm = w / w.sum()
    s_norm = s/s.sum()
    denominator = torch.mul(s_norm, w_norm)
    denominator = denominator.sum()

    # disjunction numberator
    r = X
    numerator = torch.mul(s_norm, w_norm)
    numerator = torch.mul(numerator, r)
    numerator = numerator.sum()
    robust = numerator/denominator
    return robust

def loss_function(Y, y_hat):
    delta = 1.0
    res = torch.exp(-delta * Y * y_hat)
    return res

def STL_formula(formula):
    lexer = stlLexer(InputStream(formula))
    tokens = CommonTokenStream(lexer)
    parser = stlParser(tokens)
    t = parser.stlProperty()
    ast = STLAbstractSyntaxTreeExtractor().visit(t)
    return ast


if __name__ == '__main__':
    # define two formulas
    formula1 = "F[0, 2] x > 2"
    formula2 = "F[0, 2] x < -2"
    ast1 = STL_formula(formula1)
    ast2 = STL_formula(formula2)

    # compute robust degrees
    varnames = ['x']
    data = [[0, 3, 2]]
    timepoints = [i for i in range(len(data[0]))]
    s = Trace(varnames, timepoints, data)
    robust1 =  ast1.robustness(s, 0, 20)
    robust2 =  ast2.robustness(s, 0, 20)
    print('r1:', robust1, 'r2:', robust2)

    # initial random weight
    w = torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)
    X = torch.tensor([[robust1, robust2]], dtype=torch.float32)

    # label of the wSTL for the data
    Y = torch.tensor([1], dtype=torch.float32)

    y_hat = forward_conjunction(X, w)

    l = loss_function(Y[0], y_hat)

    print('robustness before training: forward_disjunction(X)= {}'.format(y_hat))


    learning_rate = 0.01
    optimizer = torch.optim.SGD([w], lr=learning_rate)
    n_iters = 5000

    for epoch in range(n_iters):
        y_hat = forward_disjunction(X, w)

        l = loss_function(Y, y_hat)

        # # backward pass for gradient
        l.backward(retain_graph=True)  #dl/dw
        #
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            #w -= learning_rate * w.grad
            for i in range(len(w)):
                if w[i] < 0.0:
                    w[i] = 0.0
        if epoch % 100 ==0:
            print('epoch {epoch}: w = {weight}, loss = {loss}'.format(epoch=epoch+1,weight=w, loss=l))

    print('robustness after training: forward_conjunction(X)= {}'.format(forward_disjunction(X,w)))
