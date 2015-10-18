from .config import is_theano, is_cgt#mode, THEANO, CGT
if is_theano():
    import theano
    import theano.tensor as T
else:
    import cgt

def grad(cost, wrt):
    if is_theano():
        return theano.gradient.grad(cost, wrt)
    else:
        return cgt.grad(cost, wrt)

def grad_clip(x, lower_bound, upper_bound):
    if is_theano():
        return theano.gradient.grad_clip(x, lower_bound, upper_bound)
    else:
        #print "Warning: grad_clip ignored in CGT"
        return x
