import tensorflow as tf
from tensorfuse.tensorflow.compat import tf_var_from_shape, tf_method_wrapper, get_raw_dimensions
from tensorfuse.tensorflow.compat import tf_method, tf_get_session, tf_ensure_init_variables, tf_property_getter
from tensorflow.python.framework import ops


class TensorType(object):
    def __init__(self, dtype, broadcastable, name=None, sparse_grad=False):
        if sparse_grad:
            raise NotImplementedError
        self.dtype = dtype
        self.broadcastable = broadcastable
        if name is not None and ':' in name:
            name = name.split(':')[0]
        self.name = name

    def __call__(self, name=None):
        if name is None:
            name = self.name
        elif ':' in name:
            name = name.split(':')[0]
        return tf_var_from_shape(name, None, self.dtype, len(self.broadcastable))


def matrix(name, dtype=None, fixed_shape=None):
    return tf_var_from_shape(name, fixed_shape, dtype, ndim=2)


def vector(name, dtype=None, fixed_shape=None):
    return tf_var_from_shape(str(name), fixed_shape, dtype, ndim=1)


def ivector(name, fixed_shape=None):
    return vector(name, 'int32', fixed_shape)


def tensor4(name, dtype=None, fixed_shape=None):
    return tf_var_from_shape(name, fixed_shape, dtype, ndim=4)


def scalar(name):
    return tf.Variable(0, name=name)


def mean(x):
    return tf.reduce_mean(x)


def tile(x, reps, ndim=None):
    if ndim is not None and ndim != x.ndim:
        raise NotImplementedError
    if isinstance(reps, tf.Tensor):
        return tf.tile(x, reps)
    else:
        return tf.tile(x, tf.pack(reps))


def maximum(x, y):
    return tf.maximum(x, y)


def exp(x):
    result = tf.exp(x)
    if get_raw_dimensions(result).ndims is None:
        result.set_shape(get_raw_dimensions(x))
    return result


def switch(x, a, b):
    x_b = tf.cast(x, 'bool')
    a_shape = a.shape
    x_shape = x.shape
    tile_cnt = [a_shp / x_shp for a_shp, x_shp in zip(a_shape, x_shape)]
    x_tiled = tf.tile(x_b, tf.pack(tile_cnt))
    return tf.select(x_tiled, a, b)


def square(x):
    return tf.square(x)


sqr = square


def sum(x, axis=None):
    if isinstance(x, list):
        x = tf.pack(x)
    if axis is None:
        result = tf.reduce_sum(x)
        result_shape = []
    else:
        if axis < 0:
            axis = x.ndim + axis
        result = tf.reduce_sum(x, axis)
        result_shape = get_raw_dimensions(x)
        result_shape = list(result_shape[:axis]) + list(result_shape[axis + 1:])
    if get_raw_dimensions(result).ndims is None:
        result.set_shape(result_shape)
    return result


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
    if isinstance(shp, tuple):
        shp = list(shp)
    if any(isinstance(s, (tf.Variable, tf.Tensor)) for s in shp):
        shp = tf.pack(shp)
    return tf.reshape(x, shp)


def stack(tensors, axis=0):
    if axis is not 0:
        raise ValueError('only axis=0 is supported under tf')
    return tf.pack(tensors)


def concatenate(items, axis=0):
    items = [tf.pack(x) if isinstance(x, list) else x for x in items]
    return tf.concat(concat_dim=axis, values=items)


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
        return tf.ones(tf.pack(shape))
    else:
        return tf.ones([shape])


def cast(x, dtype):
    return tf.cast(x, dtype)


def flatten(x, outdim=1):
    if outdim == 1:
        return tf.reshape(x, [-1])
    else:
        return tf.reshape(x, [x.shape[0], -1])


def max(x):
    return tf.reduce_max(x)


def eq(x, y):
    return tf.equal(x, y)


def argmax(x, axis):
    return tf.argmax(x, dimension=axis)


@tf_method([tf.Variable], "dimshuffle")
def _tf_variable_dimshuffle(self, *args, **kwargs):
    return dimshuffle(self, *args, **kwargs)


@tf_method([tf.Variable], "reshape")
def _tf_variable_reshape(self, *args, **kwargs):
    return reshape(self, *args, **kwargs)


@tf_method([tf.Variable, tf.Tensor], "flatten")
def _tf_variable_flatten(self, *args, **kwargs):
    return flatten(self, *args, **kwargs)


@tf_method([tf.Variable], "get_value")
def _tf_variable_get_value(self, borrow=None):
    tf_ensure_init_variables()
    return tf_get_session().run(self)


@tf_method([tf.Variable], "set_value")
def _tf_variable_set_value(self, val, borrow=None):
    tf_get_session().run(tf.assign(self, val))


@tf_method([tf.Variable, tf.Tensor], "astype")
def _tf_astype(self, dtype):
    if isinstance(dtype, str):
        return tf.cast(self, dtype)
    return tf.cast(self, dtype.base_dtype)


@tf_property_getter([tf.Variable], "broadcastable")
def _tf_variable_broadcastable(self):
    return [True] * len(self.shape)


_old_sub = getattr(ops.Tensor, "__sub__")


@tf_method([tf.Variable, tf.Tensor], "__sub__")
def _tf_sub(self, other):
    self_dim = list(get_raw_dimensions(self))
    other_dim = list(get_raw_dimensions(other))

    if isinstance(self, tf.Variable):
        self = self._AsTensor()
    if isinstance(other, tf.Variable):
        other = other._AsTensor()
    result = _old_sub(self, other)
    result_dim = get_raw_dimensions(result)
    if result_dim.ndims is None:
        # we could infer the shape in this case
        if len(self_dim) > len(other_dim):
            result.set_shape(self_dim)
        else:
            result.set_shape(other_dim)
    return result


_old_mul = getattr(ops.Tensor, "__mul__")


@tf_method([tf.Variable, tf.Tensor], "__mul__")
def _tf_mul(self, other):
    self_dim = get_raw_dimensions(self)
    other_dim = get_raw_dimensions(other)
    if isinstance(self, tf.Variable):
        self = self._AsTensor()
    if isinstance(other, tf.Variable):
        other = other._AsTensor()
    if not self.dtype.is_floating and (isinstance(other, float) or other.dtype.is_floating):
        self = tf.cast(self, tf.float32)
    if self_dim.ndims is not None and other_dim.ndims is not None:
        result = _old_mul(self, other)
        result_dim = get_raw_dimensions(result)
        if result_dim.ndims is None:
            # we could infer the shape in this case
            if len(self_dim) > len(other_dim):
                result.set_shape(self_dim)
            else:
                result.set_shape(other_dim)
        return result
    else:
        return _old_mul(self, other)


_old_rmul = getattr(ops.Tensor, "__rmul__")


@tf_method([tf.Variable, tf.Tensor], "__rmul__")
def _tf_rmul(self, other):
    self_dim = list(get_raw_dimensions(self))
    other_dim = list(get_raw_dimensions(other))

    if isinstance(self, tf.Variable):
        self = self._AsTensor()
    if isinstance(other, tf.Variable):
        other = other._AsTensor()
    if not self.dtype.is_floating and (isinstance(other, float) or other.dtype.is_floating):
        self = tf.cast(self, tf.float32)
    result = _old_rmul(self, other)
    result_dim = get_raw_dimensions(result)
    if result_dim.ndims is None:
        # we could infer the shape in this case
        if len(self_dim) > len(other_dim):
            result.set_shape(self_dim)
        else:
            result.set_shape(other_dim)
    return result
