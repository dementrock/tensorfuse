import cgt


def grad(cost, wrt, known_grads=None):
    if known_grads:
        raise ValueError('cgt does not support known_grads yet')
    return cgt.grad(cost, wrt)
