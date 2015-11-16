import theano
import theano.tensor as T
import theano.tensor.nlinalg

def matrix_inverse(x):
    return theano.tensor.nlinalg.matrix_inverse(x)
