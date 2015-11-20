import theano
import theano.tensor as T


def matrix(name, dtype=None, fixed_shape=None):
    # Ignore fixed shape
    return T.matrix(name, dtype)


def imatrix(name):
    return T.imatrix(name)


def ivector(name):
    return T.ivector(name)


def col(name):
    return T.col(name)


def icol(name):
    return T.icol(name)


def vector(name, dtype=None, fixed_shape=None):
    return T.vector(name, dtype)


def scalar(name):
    return T.scalar(name)


def mean(x):
    return T.mean(x)


def tile(x, reps):
    return T.tile(x, reps)


def switch(x, a, b):
    return T.switch(x, a, b)


def square(x):
    return T.square(x)


sqr = square


def log(x):
    return T.log(x)


def exp(x):
    return T.exp(x)


def prod(*args, **kwargs):
    import ipdb; ipdb.set_trace()


def sum(x, axis=None):
    return T.sum(x, axis=axis)


def dot(x, y):
    return T.dot(x, y)


def dimshuffle(x, *pattern):
    return x.dimshuffle(*pattern)


def tanh(x):
    return T.tanh(x)


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
    _ensure_broadcastable(a, b, bcpat)
    if x == '+':
        return a + b
    if x == '*':
        return a * b
    import ipdb; ipdb.set_trace()


def reshape(x, shp):
    return T.reshape(x, shp)


def stack(tensors, axis=0):
    return T.stack(tensors, axis=axis)


def hstack(tensors):
    return T.horizontal_stack(*tensors)


def vstack(tensors):
    return T.vertical_stack(*tensors)


def ones(shape):
    return T.ones(shape)


def ones_like(x):
    return T.ones_like(x)


def concatenate(items, axis=0):
    return T.concatenate(items, axis=axis)


def sqrt(x):
    return T.sqrt(x)


def constant(x):
    return T.constant(x)


def max(x, axis=None, keepdims=False):
    return T.max(x, axis=axis, keepdims=keepdims)


def arange(x):
    return T.arange(x)


def minimum(x, y):
    return T.minimum(x, y)


def abs(x):
    return x.__abs__()

def sin(x):
    return T.sin(x)


def cos(x):
    return T.cos(x)


def clip(x, low, high):
    return T.clip(x, low, high)


def take(x, indices, axis=None):
    return T.take(x, indices, axis=axis)


def std(x):
    return T.std(x)


def argmax(x, axis=None, keepdims=False):
    return T.argmax(x, axis=axis, keepdims=keepdims)


def argmin(x, axis=None, keepdims=False):
    return T.argmin(x, axis=axis, keepdims=keepdims)


def eq(x, y):
    return T.eq(x, y)


def neq(x, y):
    return T.neq(x, y)


def diag(x):
    return T.diag(x)


def mod(x, y):
    return T.mod(x, y)


def power(x, n):
    return T.power(x, n)


def zeros(shape):
    return T.zeros(shape)


def cast(x, dtype):
    return T.cast(x, dtype)


def outer(x, y):
    return T.outer(x, y)


def eye(n):
    return T.eye(n)
