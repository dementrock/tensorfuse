from .config import is_theano
if is_theano():
    import theano
    import theano.tensor as T
else:
    import cgt

def is_shared(x):
    """
    Check if x is a SharedVariable in Theano, or created via cgt.shared in CGT
    """
    if is_theano():
        return isinstance(x, theano.compile.SharedVariable)
    else:
        if isinstance(x, cgt.core.Node):
            import ipdb; ipdb.set_trace()
        else:
            return False

def tensor(dtype, ndim, name=None, fixed_shape=None):
    """
    Create tensor variable from given information
    """
    if is_theano():
        if fixed_shape is not None:
            print 'fixed shape ignored in Theano'
        return T.TensorType(dtype, [False] * ndim)
    else:
        return cgt.tensor(dtype, ndim, name, fixed_shape)

def get_value(x, borrow=None):
    """
    Get parameter value from a shared variable.
    """
    if is_theano():
        borrow = borrow or False
        return x.get_value(borrow=borrow)
    else:
        return x.op.get_value()

def set_value(x, val):
    """
    Get parameter value from a shared variable.
    """
    if is_theano():
        x.set_value(val)
    else:
        return x.op.get_value()
