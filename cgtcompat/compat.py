from .config import is_theano, is_cgt, is_tf
from utils import wrap_into_list
if is_theano():
    import theano
    import theano.tensor as T
elif is_cgt():
    import cgt
else:
    import tensorflow as tf
import numpy as np

def is_shared(x):
    """
    Check if x is a SharedVariable in Theano, or created via cgt.shared in CGT
    """
    if is_theano():
        return isinstance(x, theano.compile.SharedVariable)
    elif is_cgt():
        return isinstance(x, cgt.core.Node) and isinstance(x.op, cgt.core.InMemoryData)
    else:
        return hasattr(x, '_cgtcompat_shared')
        

# In Theano, TensorVariable and SharedVariable are different, and they do not
# inherit from each other
def is_tensor(x):
    if is_theano():
        return isinstance(x, T.TensorVariable)
    elif is_cgt():
        return isinstance(x, cgt.core.Node) and x.is_tensor() and not isinstance(x.op, cgt.core.InMemoryData)
    else:
        return isinstance(x, tf.Variable)

def tensor(dtype, ndim, name=None, fixed_shape=None):
    """
    Create tensor variable from given information
    """
    if is_theano():
        if fixed_shape is not None:
            print 'fixed shape ignored in Theano'
        return T.TensorType(dtype, [False] * ndim)(name)
    elif is_cgt():
        return cgt.tensor(dtype, ndim, name, fixed_shape)
    else:
        return tf_var_from_shape(name, fixed_shape, dtype, ndim)

def get_value(x, borrow=None):
    """
    Get parameter value from a shared variable.
    """
    if is_theano():
        borrow = borrow or False
        return x.get_value(borrow=borrow)
    elif is_cgt():
        return x.op.get_value()
    else:
        tf_ensure_init_variables()
        return x.eval()

def set_value(x, val):
    """
    Get parameter value from a shared variable.
    """
    if is_theano():
        x.set_value(val)
    elif is_cgt():
        x.op.set_value(val)
    else:
        import ipdb; ipdb.set_trace()

# check if some object is a node in the computation graph (either a variable, or an operation on variables)
def is_variable(x):
    if is_theano():
        return isinstance(x, theano.gof.Variable)
    elif is_cgt():
        return isinstance(x, cgt.core.Node)
    elif is_tf():
        return isinstance(x, (tf.Tensor, tf.Variable))
    else:
        import ipdb; ipdb.set_trace()


def broadcastable(x):
    if is_theano():
        return x.broadcastable
    else:
        return None

def get_inputs(outputs):
    if is_theano():
        return theano.gof.graph.inputs(outputs)
    else:
        outputs = list(outputs)
        return [node for node in cgt.core.topsorted(outputs) if node.is_input()]

def shape(x):
    if is_theano():
        return x.shape
    elif is_cgt():
        return x.shape
    elif is_tf():
        if isinstance(x, tf.Tensor):
            return x._shape
        elif isinstance(x, tf.Variable):
            return x._initial_value._shape
        else:
            import ipdb; ipdb.set_trace()
    else:
        import ipdb; ipdb.set_trace()


if is_tf():
    _tf_blank_vars = []

    def tf_ensure_init_variables():
        tf.initialize_variables(_tf_blank_vars).run()
        del _tf_blank_vars[:]

    def tf_add_blank_var(var):
        _tf_blank_vars.append(var)

    def tf_var_from_shape(name, fixed_shape, dtype, ndim):
        if fixed_shape and all(fixed_shape):
            var = tf.Variable(np.zeros(fixed_shape), name=name)
            var._cgtcompat_initialized = False
            var._cgtcompat_shared = False
            tf_add_blank_var(var)
            return var
        else:
            #raise ValueError('shape must be specified under tensorflow')
            fixed_shape = fixed_shape or [None] * ndim
            dtype = dtype or floatX
            var = tf.Variable(tf.zeros([]), name=name, validate_shape=False)
            var._cgtcompat_initialized = False
            var._cgtcompat_shared = False
            tf_add_blank_var(var)
            return var

    def _tf_tensor_ndim(self):
        return len(self._shape)

    def _tf_tensor_shape(self):
        return map(lambda x: x.value, self._shape)

    def _tf_variable_ndim(self):
        return len(self._initial_value._shape)

    def _tf_variable_shape(self):
        return map(lambda x: x.value, self._initial_value._shape)

    def _tf_pow(self, x):
        if x == 2:
            return tf.square(self)
        return tf.pow(self, x)

    def _mk_tf_getitem(old_getitem):
        def getitem(self, slices):
            try:
                if not isinstance(slices, tuple):
                    slices = (slices,)
                if len(slices) < self.ndim:
                    slices = slices + (slice(None),) * (self.ndim - len(slices))
                rev_dims = [idx for idx, x in enumerate(slices) if isinstance(x, slice) and x.step == -1]
                if len(rev_dims) == 1:
                    rev_dim = rev_dims[0]
                    return tf.reverse(self, [False] * (rev_dim) + [True] + [False] * (self.ndim - rev_dim - 1))
                return old_getitem(self, slices)
            except Exception:
                import ipdb; ipdb.set_trace()
        return getitem


    tf.Variable.ndim = property(_tf_variable_ndim, "ndim")
    tf.Variable.shape = property(_tf_variable_shape, "shape")
    tf.Variable.__getitem__ = _mk_tf_getitem(tf.Variable.__getitem__)
    tf.Tensor.ndim = property(_tf_tensor_ndim, "ndim")
    tf.Tensor.shape = property(_tf_tensor_shape, "shape")
    tf.Tensor.__getitem__ = _mk_tf_getitem(tf.Tensor.__getitem__)

    tf.Variable.__pow__ = _tf_pow
    tf.Tensor.__pow__ = _tf_pow
