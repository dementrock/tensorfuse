from .config import is_theano, is_cgt#mode, THEANO, CGT
if is_theano():
    import theano
    import theano.tensor as T
else:
    import cgt

def grad(cost, wrt, known_grads=None):
    if is_theano():
        return theano.gradient.grad(cost, wrt, known_grads=known_grads)
    else:
        if known_grads:
            raise ValueError('cgt does not support known_grads yet')
        return cgt.grad(cost, wrt)

def grad_clip(x, lower_bound, upper_bound):
    if is_theano():
        return theano.gradient.grad_clip(x, lower_bound, upper_bound)
    else:
        #print "Warning: grad_clip ignored in CGT"
        return x
