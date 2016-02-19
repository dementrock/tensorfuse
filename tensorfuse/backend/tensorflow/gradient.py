import tensorflow as tf
from tensorfuse.utils import format_as


def grad(cost, wrt, known_grads=None):
    ret = tf.gradients(cost, wrt)
    if isinstance(wrt, list):
        return [x if x is not None else tf.zeros_like(wrt[i]) for i, x in enumerate(ret)]
    else:
        return ret[0] or tf.zeros_like(wrt)


def grad_clip(x, lb, ub):
    return x


def jacobian(expression, wrt):
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
