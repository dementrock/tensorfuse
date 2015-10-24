from ..config import is_theano, is_cgt
if is_theano():
    import theano
    import theano.tensor as T
else:
    import cgt

def matrix(name, dtype=None, fixed_shape=None):
    if is_theano():
        return T.matrix(name, dtype)
    else:
        return cgt.matrix(name, dtype, fixed_shape)

def imatrix(name):
    if is_theano():
        return T.imatrix(name)
    else:
        return cgt.matrix(name, dtype='int32')

def ivector(name):
    if is_theano():
        return T.ivector(name)
    else:
        return cgt.vector(name, dtype='int32')

def col(name):
    if is_theano():
        return T.col(name)
    else:
        return cgt.matrix(name, fixed_shape=(None, 1))

def icol(name):
    if is_theano():
        return T.icol(name)
    else:
        return cgt.matrix(name, dtype='int32', fixed_shape=(None, 1))


def vector(name):
    if is_theano():
        return T.vector(name)
    else:
        return cgt.vector(name)

def scalar(name):
    if is_theano():
        return T.scalar(name)
    else:
        return cgt.scalar(name)

def mean(x):
    if is_theano():
        return T.mean(x)
    else:
        return cgt.mean(x)

def tile(x, reps):
    if is_theano():
        return T.tile(x, reps)
    else:
        out = x
        for i, nrep in enumerate(reps):
            if nrep > 1:
                out = cgt.repeat(out, nrep, axis=i)
        return out

def square(x):
    if is_theano():
        return T.square(x)
    else:
        return cgt.square(x)

def log(x):
    if is_theano():
        return T.log(x)
    else:
        return cgt.log(x)

def exp(x):
    if is_theano():
        return T.exp(x)
    else:
        return cgt.exp(x)

def prod(*args, **kwargs):
    import ipdb; ipdb.set_trace()

def sum(x, axis=None):
    if is_theano():
        return T.sum(x, axis=axis)
    else:
        return cgt.sum(x, axis=axis)

def dot(x, y):
    if is_theano():
        return T.dot(x, y)
    else:
        return cgt.dot(x, y)

def dimshuffle(x, *pattern):
    if is_theano():
        return x.dimshuffle(*pattern)
    else:
        return cgt.dimshuffle(x, list(pattern))

def tanh(x):
    if is_theano():
        return theano.tensor.tanh(x)
    else:
        return cgt.tanh(x)

def _ensure_broadcastable(a, b, bcpat):
    x, y = a, b
    xpat, ypat = bcpat.split(',')
    xpat = xpat.strip()
    ypat = ypat.strip()
    for i, xent in enumerate(xpat):
        if xent == '1' and not a.broadcastable[i]:
            raise ValueError('The %dth dimension of %s is not broadcastable', i, str(a))
    for i, yent in enumerate(ypat):
        if yent == '1' and not b.broadcastable[i]:
            raise ValueError('The %dth dimension of %s is not broadcastable', i, str(b))

def broadcast(x, a, b, bcpat):
    if is_theano():
        _ensure_broadcastable(a, b, bcpat)
        if x == '+':
            return a + b
        if x == '*':
            return a * b
        import ipdb; ipdb.set_trace()
    else:
        return cgt.broadcast(x, a, b, bcpat)

def reshape(x, shp):
    if is_theano():
        return T.reshape(x, shp)
    else:
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
    if is_theano():
        return T.stack(tensors, axis=axis)
    else:
        if axis is not 0:
            raise ValueError('only axis=0 is supported under cgt')
        return cgt.concatenate(map(lambda x: cgt.reshape(x, [1] + x.shape), tensors), axis=0)

def ones(shape):
    if is_theano():
        return T.ones(shape)
    else:
        return cgt.ones(shape)

def ones_like(x):
    if is_theano():
        return T.ones_like(x)
    else:
        return cgt.ones_like(x)

def concatenate(items, axis=0):
    if is_theano():
        return T.concatenate(items, axis=axis)
    else:
        return cgt.concatenate(items, axis=axis)

def sqrt(x):
    if is_theano():
        return T.sqrt(x)
    else:
        return cgt.sqrt(x)

def constant(x):
    if is_theano():
        return T.constant(x)
    else:
        return cgt.constant(x)

def max(x, axis=None, keepdims=False):
    if is_theano():
        return T.max(x, axis=axis, keepdims=keepdims)
    else:
        return cgt.max(x, axis=axis, keepdims=keepdims)

def arange(x):
    if is_theano():
        return T.arange(x)
    else:
        return cgt.arange(x)

def minimum(x, y):
    if is_theano():
        return T.minimum(x, y)
    else:
        return x * (x < y) + y * (x >= y)

def abs(x):
    if is_theano():
        return x.__abs__()
    else:
        return cgt.core.Result(cgt.core.ElwiseUnary("abs"), [x])
