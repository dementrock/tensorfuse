from .config import is_theano, is_cgt#mode, THEANO, CGT
if is_theano():
    import theano
    import theano.tensor as T
elif is_cgt():
    import cgt
else:
    import tensorflow as tf

def grad(cost, wrt, known_grads=None):
    if is_theano():
        return theano.gradient.grad(cost, wrt, known_grads=known_grads)
    elif is_cgt():
        if known_grads:
            raise ValueError('cgt does not support known_grads yet')
        return cgt.grad(cost, wrt)
    else:
        if isinstance(wrt, list):
            return tf.gradients(cost, wrt)
        else:
            return tf.gradients(cost, wrt)[0]

def grad_clip(x, lower_bound, upper_bound):
    if is_theano():
        return theano.gradient.grad_clip(x, lower_bound, upper_bound)
    else:
        #print "Warning: grad_clip ignored in CGT"
        return x
