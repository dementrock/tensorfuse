import cgt

def matrix(name, dtype=None, fixed_shape=None):
    return cgt.matrix(name, dtype, fixed_shape)


def imatrix(name):
    return cgt.matrix(name, dtype='int32')


def ivector(name):
    return cgt.vector(name, dtype='int32')


def col(name):
    return cgt.matrix(name, fixed_shape=(None, 1))


def icol(name):
    return cgt.matrix(name, dtype='int32', fixed_shape=(None, 1))


def vector(name, dtype=None, fixed_shape=None):
    return cgt.vector(name, dtype, fixed_shape)


def scalar(name):
    return cgt.scalar(name)


def mean(x):
    return cgt.mean(x)


def tile(x, reps):
    out = x
    for i, nrep in enumerate(reps):
        if nrep > 1:
            out = cgt.repeat(out, nrep, axis=i)
    return out


def square(x):
    return cgt.square(x)


sqr = square


def log(x):
    return cgt.log(x)


def exp(x):
    return cgt.exp(x)


def sum(x, axis=None):
    return cgt.sum(x, axis=axis)


def dot(x, y):
    return cgt.dot(x, y)


def dimshuffle(x, *pattern):
    return cgt.dimshuffle(x, list(pattern))


def tanh(x):
    return cgt.tanh(x)


def broadcast(x, a, b, bcpat):
    return cgt.broadcast(x, a, b, bcpat)


def reshape(x, shp):
    from ..utils import wrap_into_tuple
    import operator
    shp = wrap_into_tuple(shp)
    neg_indices = [(idx, shp_slice) for idx, shp_slice in enumerate(shp) if shp_slice == -1]
    if len(neg_indices) > 1:
        raise ValueError('At most one reshaped dimension can be -1')
    elif len(neg_indices) == 1:
        idx, shp_slice = neg_indices[0]
        neg_size = reduce(operator.mul, x.shape, 1) // reduce(operator.mul, shp[:idx] + shp[idx+1:], 1)
        shp = tuple(shp[:idx] + (neg_size,) + shp[idx+1:])
        return cgt.reshape(x, shp)
    else:
        return cgt.reshape(x, shp)


def stack(tensors, axis=0):
    if axis is not 0:
        raise ValueError('only axis=0 is supported under cgt')
    return cgt.concatenate(map(lambda x: cgt.reshape(x, [1] + x.shape), tensors), axis=0)


def hstack(tensors):
    return cgt.vstack(tensors)


def vstack(tensors):
    return cgt.vstack(tensors)


def ones(shape):
    return cgt.ones(shape)


def ones_like(x):
    return cgt.ones_like(x)


def concatenate(items, axis=0):
    return cgt.concatenate(items, axis=axis)


def sqrt(x):
    return cgt.sqrt(x)


def constant(x):
    return cgt.constant(x)


def max(x, axis=None, keepdims=False):
    return cgt.max(x, axis=axis, keepdims=keepdims)


def arange(x):
    return cgt.arange(x)


def minimum(x, y):
    return x * (x < y) + y * (x >= y)


def abs(x):
    return cgt.core.Result(cgt.core.ElwiseUnary("abs"), [x])


def sin(x):
    return cgt.sin(x)


def cos(x):
    return cgt.cos(x)


def std(x):
    return cgt.std(x)


def argmax(x, axis=None, keepdims=False):
    return cgt.argmax(x, axis=axis, keepdims=keepdims)


def argmin(x, axis=None, keepdims=False):
    return cgt.argmin(x, axis=axis, keepdims=keepdims)


def eq(x, y):
    return cgt.eq(x, y)


def neq(x, y):
    return cgt.neq(x, y)


def diag(x):
    return cgt.diag(x)
