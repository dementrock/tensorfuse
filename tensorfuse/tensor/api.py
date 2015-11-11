from tensorfuse.config import is_theano, is_cgt, is_tf
from tensorfuse.gradient import grad
if is_theano():
    import theano
    import theano.tensor as T
elif is_cgt():
    import cgt
else:
    import tensorflow as tf
    from tensorfuse.compat import tf_var_from_shape

import numpy as np

def matrix(name, dtype=None, fixed_shape=None):
    if is_theano():
        return T.matrix(name, dtype)
    elif is_cgt():
        return cgt.matrix(name, dtype, fixed_shape)
    else:
        return tf_var_from_shape(name, fixed_shape, dtype, ndim=2)


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


def vector(name, dtype=None, fixed_shape=None):
    if is_theano():
        return T.vector(name, dtype)
    elif is_cgt():
        return cgt.vector(name, dtype, fixed_shape)
    else:
        return tf_var_from_shape(name, fixed_shape, dtype, ndim=1)

def scalar(name):
    if is_theano():
        return T.scalar(name)
    elif is_cgt():
        return cgt.scalar(name)
    else:
        return tf.Variable(0, name=name)

def mean(x):
    if is_theano():
        return T.mean(x)
    elif is_cgt():
        return cgt.mean(x)
    elif is_tf():
        return tf.reduce_mean(x)
    else:
        import ipdb; ipdb.set_trace()

def tile(x, reps):
    if is_theano():
        return T.tile(x, reps)
    elif is_cgt():
        out = x
        for i, nrep in enumerate(reps):
            if nrep > 1:
                out = cgt.repeat(out, nrep, axis=i)
        return out
    elif is_tf():
        if isinstance(reps, tf.Tensor):
            return tf.tile(x, reps)
        else:
            return tf.tile(x, tf.pack(reps))
    else:
        import ipdb; ipdb.set_trace()

def switch(x, a, b):
    if is_theano():
        return T.switch(x, a, b)
    elif is_cgt():
        import ipdb; ipdb.set_trace()
    elif is_tf():
        return tf.select(x, a, b)
    else:
        import ipdb; ipdb.set_trace()

def square(x):
    if is_theano():
        return T.square(x)
    elif is_cgt():
        return cgt.square(x)
    else:
        return tf.square(x)

sqr = square

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
    elif is_cgt():
        return cgt.sum(x, axis=axis)
    elif is_tf():
        return tf.reduce_sum(x, axis)
    else:
        import ipdb; ipdb.set_trace()

def dot(x, y):
    if is_theano():
        return T.dot(x, y)
    elif is_cgt():
        return cgt.dot(x, y)
    elif is_tf():
        if x.ndim == 2 and y.ndim == 2:
            return tf.matmul(x, y)
        elif x.ndim == 1 and y.ndim == 1:
            return tf.reduce_sum(x * y)
        elif x.ndim == 1 and y.ndim == 2:
            return tf.reshape(tf.matmul(tf.expand_dims(x, 0), y), [-1])
        elif x.ndim == 2 and y.ndim == 1:
            return tf.reshape(tf.matmul(x, tf.expand_dims(y, 1)), [-1])
        else:
            import ipdb; ipdb.set_trace()
    else:
        import ipdb; ipdb.set_trace()

def dimshuffle(x, *pattern):
    if is_theano():
        return x.dimshuffle(*pattern)
    elif is_cgt():
        return cgt.dimshuffle(x, list(pattern))
    else:
        # First, get rid of all occurrences of 'x'
        pure_pattern = [p for p in pattern if p != 'x']
        dims = range(x.ndim)
        perm = list(pure_pattern) + sorted(set(dims) - set(pure_pattern))
        res = tf.transpose(x, perm=perm)
        # For each occurrence of 'x', apply expand_dims
        x_indices = [idx for idx, p in enumerate(pattern) if p == 'x']
        if len(x_indices) == 0:
            return res
        elif len(x_indices) > 1:
            # too lazy for now
            import ipdb; ipdb.set_trace()
        else:
            return tf.expand_dims(res, x_indices[0])

def tanh(x):
    if is_theano():
        return theano.tensor.tanh(x)
    elif is_cgt():
        return cgt.tanh(x)
    elif is_tf():
        return tf.tanh(x)
    else:
        import ipdb; ipdb.set_trace()

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
    elif is_cgt():
        return cgt.broadcast(x, a, b, bcpat)
    else:
        if x == '+':
            return a + b
        if x == '*':
            return a * b
        import ipdb; ipdb.set_trace()


def reshape(x, shp):
    if is_theano():
        return T.reshape(x, shp)
    elif is_cgt():
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
    elif is_tf():
        return tf.reshape(x, shp)
    else:
        import ipdb; ipdb.set_trace()


def stack(tensors, axis=0):
    if is_theano():
        return T.stack(tensors, axis=axis)
    elif is_cgt():
        if axis is not 0:
            raise ValueError('only axis=0 is supported under cgt')
        return cgt.concatenate(map(lambda x: cgt.reshape(x, [1] + x.shape), tensors), axis=0)
    elif is_tf():
        if axis is not 0:
            raise ValueError('only axis=0 is supported under tf')
        return tf.pack(tensors)
    else:
        import ipdb; ipdb.set_trace()

def hstack(tensors):
    if is_theano():
        return T.horizontal_stack(*tensors)
    else:
        return cgt.vstack(tensors)


def vstack(tensors):
    if is_theano():
        return T.vertical_stack(*tensors)
    else:
        return cgt.vstack(tensors)


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
    elif is_cgt():
        return cgt.concatenate(items, axis=axis)
    elif is_tf():
        return tf.concat(concat_dim=axis, values=list(items))
    else:
        import ipdb; ipdb.set_trace()

def sqrt(x):
    if is_theano():
        return T.sqrt(x)
    elif is_cgt():
        return cgt.sqrt(x)
    elif is_tf():
        return tf.sqrt(x)
    else:
        import ipdb; ipdb.set_trace()

def constant(x):
    if is_theano():
        return T.constant(x)
    elif is_cgt():
        return cgt.constant(x)
    elif is_tf():
        return tf.constant(x)
    else:
        import ipdb; ipdb.set_trace()

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

def sin(x):
    if is_theano():
        return T.sin(x)
    elif is_cgt():
        return cgt.sin(x)
    elif is_tf():
        return tf.sin(x)
    else:
        import ipdb; ipdb.set_trace()

def cos(x):
    if is_theano():
        return T.cos(x)
    elif is_cgt():
        return cgt.cos(x)
    elif is_tf():
        return tf.cos(x)
    else:
        import ipdb; ipdb.set_trace()


def clip(x, low, high):
    if is_theano():
        return T.clip(x, low, high)
    elif is_cgt():
        import ipdb; ipdb.set_trace()
    elif is_tf():
        return tf.clip_by_value(x, low, high)
    else:
        import ipdb; ipdb.set_trace()

def take(x, indices, axis=None):
    if is_theano():
        return T.take(x, indices, axis=axis)
    elif is_cgt():
        import ipdb; ipdb.set_trace()
    elif is_tf():
        if isinstance(indices, (list, tuple)):
            import ipdb; ipdb.set_trace()
        else:
            if axis:
                slices = (slice(None),) * axis + (indices,) + (slice(None),) * (x.ndim - axis - 1)
                return x[slices]
            else:
                return tf.reshape(x, [-1])[indices]
    else:
        import ipdb; ipdb.set_trace()


def std(x):
    if is_theano():
        return T.std(x)
    else:
        return cgt.std(x)

def argmax(x, axis=None, keepdims=False):
    if is_theano():
        return T.argmax(x, axis=axis, keepdims=keepdims)
    else:
        return cgt.argmax(x, axis=axis, keepdims=keepdims)

def argmin(x, axis=None, keepdims=False):
    if is_theano():
        return T.argmin(x, axis=axis, keepdims=keepdims)
    else:
        return cgt.argmin(x, axis=axis, keepdims=keepdims)

def eq(x, y):
    if is_theano():
        return T.eq(x, y)
    else:
        return cgt.eq(x, y)

def neq(x, y):
    if is_theano():
        return T.neq(x, y)
    else:
        return cgt.neq(x, y)

def diag(x):
    if is_theano():
        return T.diag(x)
    elif is_cgt():
        return cgt.diag(x)
    elif is_tf():
        if x.ndim == 1:
            return tf.diag(x)
        else:
            import ipdb; ipdb.set_trace()
    else:
        import ipdb; ipdb.set_trace()

def mod(x, y):
    if is_theano():
        return T.mod(x, y)
    elif is_cgt():
        import ipdb; ipdb.set_trace()
    elif is_tf():
        return tf.mod(x, y)
    else:
        import ipdb; ipdb.set_trace()


def power(x, n):
    if is_theano():
        return T.power(x, n)
    elif is_cgt():
        import ipdb; ipdb.set_trace()
    elif is_tf():
        return tf.pow(x, n)
    else:
        import ipdb; ipdb.set_trace()

def zeros(shape):
    if is_theano():
        return T.zeros(shape)
    elif is_cgt():
        import ipdb; ipdb.set_trace()
    elif is_tf():
        if isinstance(shape, (list, tuple)):
            return tf.zeros(shape)
        else:
            return tf.zeros([shape])
    else:
        import ipdb; ipdb.set_trace()

def cast(x, dtype):
    if is_theano():
        return T.cast(x, dtype)
    elif is_cgt():
        import ipdb; ipdb.set_trace()
    elif is_tf():
        return tf.cast(x, dtype)
    else:
        import ipdb; ipdb.set_trace()
