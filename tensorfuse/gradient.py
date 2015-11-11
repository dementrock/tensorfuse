from .config import is_theano, is_cgt, is_tf
if is_theano():
    import theano
    import theano.tensor as T
elif is_cgt():
    import cgt
elif is_tf():
    import tensorflow as tf
else:
    import ipdb; ipdb.set_trace()
from tensorfuse.compat import is_variable
from tensorfuse.scan_op import scan

def grad(cost, wrt, known_grads=None):
    if is_theano():
        return theano.gradient.grad(cost, wrt, known_grads=known_grads, disconnected_inputs='warn')
    elif is_cgt():
        if known_grads:
            raise ValueError('cgt does not support known_grads yet')
        return cgt.grad(cost, wrt)
    elif is_tf():
        ret = tf.gradients(cost, wrt)
        if isinstance(wrt, list):
            return [x if x is not None else tf.zeros_like(wrt[i]) for i, x in enumerate(ret)]
        else:
            return ret[0] or tf.zeros_like(wrt)
    else:
        import ipdb; ipdb.set_trace()

def grad_clip(x, lower_bound, upper_bound):
    if is_theano():
        return theano.gradient.grad_clip(x, lower_bound, upper_bound)
    elif is_cgt():
        #print "Warning: grad_clip ignored in CGT"
        return x
    elif is_tf():
        import ipdb; ipdb.set_trace()
    else:
        import ipdb; ipdb.set_trace()

def jacobian(expression, wrt):
    if is_theano():
        return theano.gradient.jacobian(expression, wrt, disconnected_inputs='warn')
    elif is_cgt():
        import ipdb; ipdb.set_trace()
    elif is_tf():
        # copying theano's implementation, which is based on scan
        #from theano.tensor import arange
        # Check inputs have the right format
        assert is_variable(expression), \
            "tensor.jacobian expects a Variable as `expression`"
        assert expression.ndim < 2, \
            ("tensor.jacobian expects a 1 dimensional variable as "
             "`expression`. If not use flatten to make it a vector")
        assert not is_variable(expression.shape[0]), \
            "shape of the expression must be known"

        using_list = isinstance(wrt, list)
        using_tuple = isinstance(wrt, tuple)

        if isinstance(wrt, (list, tuple)):
            wrt = list(wrt)
        else:
            wrt = [wrt]

        if expression.ndim == 0:
            # expression is just a scalar, use grad
            return format_as(using_list, using_tuple, grad(expression, wrt))

        def inner_function(*args):
            idx = args[0]
            expr = args[1]
            rvals = []
            for inp in args[2:]:
                try:
                    rval = grad(expr[idx], inp)
                except Exception as e:
                    import ipdb; ipdb.set_trace()
                if rval is None:
                    import ipdb; ipdb.set_trace()
                rvals.append(rval)
            return rvals
        # Computing the gradients does not affect the random seeds on any random
        # generator used n expression (because during computing gradients we are
        # just backtracking over old values. (rp Jan 2012 - if anyone has a
        # counter example please show me)
        jacobs, updates = scan(inner_function,
                               sequences=[range(expression.shape[0])],
                               non_sequences=[expression] + wrt,
                               n_steps=expression.shape[0])
        assert not updates
        return format_as(using_list, using_tuple, jacobs)
    else:
        import ipdb; ipdb.set_trace()

# From Theano
def format_as(use_list, use_tuple, outputs):
    """
    Formats the outputs according to the flags `use_list` and `use_tuple`.
    If `use_list` is True, `outputs` is returned as a list (if `outputs`
    is not a list or a tuple then it is converted in a one element list).
    If `use_tuple` is True, `outputs` is returned as a tuple (if `outputs`
    is not a list or a tuple then it is converted into a one element tuple).
    Otherwise (if both flags are false), `outputs` is returned.
    """
    assert not (use_list and use_tuple), \
        "Both flags cannot be simultaneously True"
    if (use_list or use_tuple) and not isinstance(outputs, (list, tuple)):
        if use_list:
            return [outputs]
        else:
            return (outputs,)
    elif not (use_list or use_tuple) and isinstance(outputs, (list, tuple)):
        assert len(outputs) == 1, \
            "Wrong arguments. Expected a one element list"
        return outputs[0]
    elif use_list or use_tuple:
        if use_list:
            return list(outputs)
        else:
            return tuple(outputs)
    else:
        return outputs
