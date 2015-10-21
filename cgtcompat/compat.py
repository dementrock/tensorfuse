from .config import is_theano
from utils import wrap_into_list
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
        return isinstance(x, cgt.core.Node) and isinstance(x.op, cgt.core.InMemoryData)

# In Theano, TensorVariable and SharedVariable are different, and they do not
# inherit from each other
def is_tensor(x):
    if is_theano():
        return isinstance(x, T.TensorVariable)
    else:
        return isinstance(x, cgt.core.Node) and x.is_tensor() and not isinstance(x.op, cgt.core.InMemoryData)

def tensor(dtype, ndim, name=None, fixed_shape=None):
    """
    Create tensor variable from given information
    """
    if is_theano():
        if fixed_shape is not None:
            print 'fixed shape ignored in Theano'
        return T.TensorType(dtype, [False] * ndim)(name)
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
        x.op.set_value(val)

def is_variable(x):
    if is_theano():
        return isinstance(x, theano.gof.Variable)
    else:
        return isinstance(x, cgt.core.Node)

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
