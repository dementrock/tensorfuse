import theano


def grad(cost, wrt, known_grads=None):
    return theano.gradient.grad(cost, wrt, known_grads=known_grads, disconnected_inputs='warn')


def grad_clip(x, lower_bound, upper_bound):
    return theano.gradient.grad_clip(x, lower_bound, upper_bound)


def jacobian(expression, wrt):
    return theano.gradient.jacobian(expression, wrt, disconnected_inputs='warn')


def hessian(expression, wrt):
    return theano.gradient.hessian(expression, wrt, disconnected_inputs='warn')
