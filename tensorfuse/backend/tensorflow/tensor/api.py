import tensorflow as tf
from tensorfuse.compat import tf_var_from_shape


def matrix(name, dtype=None, fixed_shape=None):
    return tf_var_from_shape(name, fixed_shape, dtype, ndim=2)


def vector(name, dtype=None, fixed_shape=None):
    return tf_var_from_shape(name, fixed_shape, dtype, ndim=1)


def ivector(name, fixed_shape=None):
    return vector(name, 'int32', fixed_shape)


def tensor4(name, dtype=None, fixed_shape=None):
    return tf_var_from_shape(name, fixed_shape, dtype, ndim=4)


def scalar(name):
    return tf.Variable(0, name=name)


def mean(x):
    return tf.reduce_mean(x)


def tile(x, reps):
    if isinstance(reps, tf.Tensor):
        return tf.tile(x, reps)
    else:
        return tf.tile(x, tf.pack(reps))


def switch(x, a, b):
    return tf.select(tf.cast(x, 'bool'), a, b)


def square(x):
    return tf.square(x)


sqr = square


def sum(x, axis=None):
    return tf.reduce_sum(x, axis)


def dot(x, y):
    if x.ndim == 2 and y.ndim == 2:
        return tf.matmul(x, y)
    elif x.ndim == 1 and y.ndim == 1:
        return tf.reduce_sum(x * y)
    elif x.ndim == 1 and y.ndim == 2:
        return tf.reshape(tf.matmul(tf.expand_dims(x, 0), y), [-1])
    elif x.ndim == 2 and y.ndim == 1:
        return tf.reshape(tf.matmul(x, tf.expand_dims(y, 1)), [-1])
    elif x.ndim == 3 and y.ndim == 2:
        x_reshaped = tf.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
        res = tf.matmul(x_reshaped, y)
        return tf.reshape(res, [x.shape[0], x.shape[1], y.shape[1]])
    else:
        import ipdb
        ipdb.set_trace()


def dimshuffle(x, *pattern):
    if isinstance(pattern[0], (list, tuple)) and len(pattern) == 1:
        pattern = pattern[0]
    # First, get rid of all occurrences of 'x'
    pure_pattern = [p for p in pattern if p != 'x']
    dims = range(x.ndim)
    perm = list(pure_pattern) + sorted(set(dims) - set(pure_pattern))
    res = tf.transpose(x, perm=perm)
    # For each occurrence of 'x', apply expand_dims
    x_indices = [idx for idx, p in enumerate(pattern) if p == 'x']
    if len(x_indices) == 0:
        return res
    # elif len(x_indices) > 1:
    #     # too lazy for now
    #     import ipdb
    #     ipdb.set_trace()
    else:
        # we need to apply the expansion backwards to make sure that the
        # indices make sense
        for ind in x_indices:
            res = tf.expand_dims(res, ind)
        return res
        # rev_inds = sorted(x_indices)[::-1]
        # return tf.expand_dims(res, x_indices[0])


def tanh(x):
    return tf.tanh(x)


def _ensure_broadcastable(a, b, bcpat):
    x, y = a, b
    xpat, ypat = bcpat.split(',')
    xpat = xpat.strip()
    ypat = ypat.strip()
    for i, xent in enumerate(xpat):
        if xent == '1' and not a.broadcastable[i]:
            raise ValueError(
                'The %dth dimension of %s is not broadcastable', i, str(a))
    for i, yent in enumerate(ypat):
        if yent == '1' and not b.broadcastable[i]:
            raise ValueError(
                'The %dth dimension of %s is not broadcastable', i, str(b))


def broadcast(x, a, b, bcpat):
    if x == '+':
        return a + b
    if x == '*':
        return a * b
    import ipdb
    ipdb.set_trace()


def reshape(x, shp):
    return tf.reshape(x, shp)


def stack(tensors, axis=0):
    if axis is not 0:
        raise ValueError('only axis=0 is supported under tf')
    return tf.pack(tensors)


def concatenate(items, axis=0):
    return tf.concat(concat_dim=axis, values=list(items))


def sqrt(x):
    return tf.sqrt(x)


def constant(x):
    return tf.constant(x)


def sin(x):
    return tf.sin(x)


def cos(x):
    return tf.cos(x)


def clip(x, low, high):
    return tf.clip_by_value(x, low, high)


def take(x, indices, axis=None):
    if isinstance(indices, (list, tuple)):
        import ipdb
        ipdb.set_trace()
    else:
        if axis:
            slices = (slice(None),) * axis + (indices,) + \
                (slice(None),) * (x.ndim - axis - 1)
            return x[slices]
        else:
            return tf.reshape(x, [-1])[indices]


def diag(x):
    if x.ndim == 1:
        return tf.diag(x)
    else:
        import ipdb
        ipdb.set_trace()


def mod(x, y):
    return tf.mod(x, y)


def power(x, n):
    return tf.pow(x, n)


def zeros(shape):
    if isinstance(shape, (list, tuple)):
        return tf.zeros(shape)
    else:
        return tf.zeros([shape])


def ones(shape):
    if isinstance(shape, (list, tuple)):
        return tf.ones(shape)
    else:
        return tf.ones([shape])


def cast(x, dtype):
    return tf.cast(x, dtype)


def flatten(x, outdim=1):
    if outdim == 1:
        return tf.reshape(x, [-1])
    else:
        return tf.reshape(x, [x.shape[0], -1])


def max():
    raise NotImplementedError


def eq(x, y):
    return tf.equal(x, y)


def argmax(x, axis):
    return tf.argmax(x, dimension=axis)
