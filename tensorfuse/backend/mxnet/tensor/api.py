import mxnet as mx


def vector(name, dtype=None, fixed_shape=None):
    return mx.symbol.Variable(name)


def sin(x):
    return mx.symbol.sin(x)
