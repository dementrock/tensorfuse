from .config import is_theano, is_cgt, is_tf, floatX
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
        return hasattr(x, '_tensorfuse_shared')
        

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
        #if fixed_shape is not None:
        #    print 'fixed shape ignored in Theano'
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
    elif is_tf():
        tf.assign(x, val).eval()
    else:
        import ipdb; ipdb.set_trace()

#no.gof.graph.inputs check if some object is a node in the computation graph (either a variable, or an operation on variables)
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
    elif is_cgt():
        outputs = list(outputs)
        return [node for node in cgt.core.topsorted(outputs) if node.is_input()]
    elif is_tf():
        outputs = list(outputs)
        return [node for node in _tf_topsorted(outputs) if _tf_is_input(node)]
    else:
        import ipdb; ipdb.set_trace()

def shape(x):
    if is_theano():
        return x.shape
    elif is_cgt():
        return x.shape
    elif is_tf():
        if isinstance(x, (tf.Tensor, tf.Variable)):
            return x.shape
        else:
            import ipdb; ipdb.set_trace()
    else:
        import ipdb; ipdb.set_trace()


if is_tf():
    _tf_blank_vars = []

    def tf_ensure_init_variables():
        if len(_tf_blank_vars) > 0:
            tf.initialize_variables(_tf_blank_vars).run()
        del _tf_blank_vars[:]

    def tf_add_blank_var(var):
        _tf_blank_vars.append(var)

    def tf_var_from_shape(name, fixed_shape, dtype, ndim):
        if fixed_shape and all(fixed_shape):
            dtype = dtype or floatX
            var = tf.Variable(np.zeros(fixed_shape, dtype=dtype), name=name)
            var._tensorfuse_shape_template = fixed_shape
            var._tensorfuse_shared = False
            tf_add_blank_var(var)
            return var
        else:
            #raise ValueError('shape must be specified under tensorflow')
            fixed_shape = fixed_shape or [None] * ndim
            nominal_shape = map(lambda x: 0 if x is None else x, fixed_shape)
            dtype = dtype or floatX
            var = tf.Variable(tf.zeros(nominal_shape, dtype=dtype), name=name, validate_shape=False)
            var._tensorfuse_shape_template = fixed_shape
            var._tensorfuse_shared = False
            tf_add_blank_var(var)
            return var

    # monkey patch a property getter to a class (or a list of classes)
    def tf_property_getter(cls_or_list, name):
        def decorator(func):
            for cls in wrap_into_list(cls_or_list):
                setattr(cls, name, property(func, name))
            return func
        return decorator

    # monkey patch a new method to a class (or a list of classes)
    def tf_method(cls_or_list, name):
        def decorator(func):
            for cls in wrap_into_list(cls_or_list):
                setattr(cls, name, func)
            return func
        return decorator

    # monkey patch an existing method. The function to be decorated should accept
    # the old method as the input, and return a wrapped method
    def tf_method_wrapper(cls_or_list, name):
        def decorator(mk_func):
            for cls in wrap_into_list(cls_or_list):
                setattr(cls, name, mk_func(getattr(cls, name)))
            return mk_func
        return decorator

    @tf_property_getter(tf.Tensor, "ndim")
    def _tf_tensor_ndim(self):
        return len(self._shape)

    @tf_property_getter([tf.Variable, tf.Tensor], "shape")
    def _tf_shape(self):
        if isinstance(self, tf.Variable):
            shape = self._initial_value._shape
        else:
            shape = self._shape
        try:
            if hasattr(self, "_tensorfuse_shape_template"):
                shape_template = self._tensorfuse_shape_template
            else:
                shape_template = shape
            none_dims = [idx for idx, size in enumerate(shape_template) if size is None]
            if len(none_dims) > 0:
                nominal_shape = map(lambda x: x.value, shape)
                # If one of the dimensions is None when the variable is created,
                # remember to return a tensor variable as opposed to the actual value
                return [x if shape_template[idx] is not None else tf.shape(self)[idx] for idx, x in enumerate(nominal_shape)]
            else:
                return map(lambda x: x.value, shape)
        except Exception as e:
            import ipdb; ipdb.set_trace()

    @tf_property_getter(tf.Variable, "ndim")
    def _tf_variable_ndim(self):
        return len(self._initial_value._shape)

    @tf_method([tf.Variable, tf.Tensor], "__pow__")
    def _tf_obj_pow(self, x):
        if x == 2:
            return tf.square(self)
        return tf.pow(self, x)

    @tf_property_getter([tf.Variable, tf.Tensor], "T")
    def _tf_obj_transpose(self):
        return tf.transpose(self)

    @tf_method([tf.Variable, tf.Tensor], "sum")
    def _tf_obj_sum(self, axis=None):
        return tf.reduce_sum(self, axis)

    # modified from cgt's topsorted code
    def _tf_topsorted(outputs):
        assert isinstance(outputs, (list,tuple))
        marks = {}
        out = []
        stack = [] #pylint: disable=W0621
        # i: node
        # jidx = number of children visited so far from that node
        # marks: state of each node, which is one of
        #   0: haven't visited
        #   1: have visited, but not done visiting children
        #   2: done visiting children
        for x in outputs:
            stack.append((x,0))
            while stack:
                (i,jidx) = stack.pop()
                if jidx == 0:
                    m = marks.get(i,0)
                    if m == 0:
                        marks[i] = 1
                    elif m == 1:
                        raise ValueError("not a dag")
                    else:
                        continue
                ps = list(i.op.inputs)
                if jidx == len(ps):
                    marks[i] = 2
                    out.append(i)
                else:
                    stack.append((i,jidx+1))
                    j = ps[jidx]
                    stack.append((j,0))
        return out

    def _tf_is_input(x):
        return len(x.op.inputs) == 0

    @tf_method_wrapper([tf.Tensor, tf.Variable], "__getitem__")
    def _mk_tf_getitem(old_getitem):
        def _fix_slice(x_size, s):
            if x_size is None or not isinstance(s, slice):
                return s
            start = s.start
            stop = s.stop
            step = s.step
            if start and start < 0:
                start = start + x_size
            if stop and stop < 0:
                stop = stop + x_size
            return slice(start, stop, step)

        def getitem(self, slices):
            try:
                target = self
                if not isinstance(slices, (list, tuple)):
                    slices = (slices,)
                if len(slices) < self.ndim:
                    slices = tuple(slices) + (slice(None),) * (self.ndim - len(slices))
                none_dims = [idx for idx, x in enumerate(slices) if x is None]
                rest_slices = [x for x in slices if x is not None]
                rest_slices = rest_slices + [slice(None,)] * (self.ndim - len(rest_slices))
                shape = self.shape
                rest_slices = [_fix_slice(shape[idx], s) for idx, s in enumerate(rest_slices)]
                rev_dims = [idx for idx, x in enumerate(rest_slices) if isinstance(x, slice) and x.start is None and x.stop is None and x.step == -1]

                # handle reverse
                if len(rev_dims) == 1:
                    rev_dim = rev_dims[0]
                    target = tf.reverse(self, [False] * (rev_dim) + [True] + [False] * (self.ndim - rev_dim - 1))
                    # after do the reversal, replace the reversed dims with a wildcard slicing
                    rest_slices[rev_dim] = slice(None)
                # handle new axis
                if len(none_dims) == 0:
                    return old_getitem(target, rest_slices)
                elif len(none_dims) == 1:
                    return tf.expand_dims(old_getitem(target, rest_slices), none_dims[0])
                else:
                    import ipdb; ipdb.set_trace()

                    
            except Exception as e:
                import ipdb; ipdb.set_trace()
        return getitem


    #tf.Variable.ndim = property(_tf_variable_ndim, "ndim")
    #tf.Variable.shape = property(_tf_variable_shape, "shape")
    #tf.Variable.__getitem__ = _mk_tf_getitem(tf.Variable.__getitem__)
    #tf.Variable.T = property(_tf_obj_transpose, "T")
    #tf.Variable.sum = _tf_obj_sum
    #tf.Tensor.ndim = property(_tf_tensor_ndim, "ndim")
    #tf.Tensor.shape = property(_tf_tensor_shape, "shape")
    #tf.Tensor.__getitem__ = _mk_tf_getitem(tf.Tensor.__getitem__)
    #tf.Tensor.T = property(_tf_obj_transpose, "T")
    #tf.Tensor.sum = _tf_obj_sum

    #tf.Variable.__pow__ = _tf_obj_pow
    #tf.Tensor.__pow__ = _tf_obj_pow

    @tf.ops.RegisterGradient("Reverse")
    def _tf_reverse_grad(op, grad):
        reverse_dims = op.inputs[1]
        return tf.array_ops.reverse(grad, reverse_dims), None
