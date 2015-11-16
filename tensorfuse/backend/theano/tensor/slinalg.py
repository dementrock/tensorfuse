import theano
import theano.tensor as T
import theano.tensor.slinalg


def solve(a, b):
    return theano.tensor.slinalg.solve(a, b)
