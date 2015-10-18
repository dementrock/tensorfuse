from ..config import is_theano, is_cgt
if is_theano():
    import theano
    import theano.tensor as T
else:
    import cgt

def matrix(name):
    if is_theano():
        return T.matrix(name)
    else:
        return cgt.matrix(name)

def vector(name):
    if is_theano():
        return T.vector(name)
    else:
        return cgt.vector(name)

def scalar(name):
    if is_theano():
        return T.scalar(name)
    else:
        return cgt.scalar(name)

def mean(a):
    if is_theano():
        return T.mean(a)
    else:
        return cgt.mean(a)

def tile(x, reps):
    if is_theano():
        return T.tile(x, reps)
    else:
        out = x
        for i, nrep in enumerate(reps):
            if nrep > 1:
                out = cgt.repeat(out, nrep, axis=i)
        return out

def square(a):
    if is_theano():
        return T.square(a)
    else:
        return cgt.square(a)

def log(a):
    if is_theano():
        return T.log(a)
    import ipdb; ipdb.set_trace()

def exp(a):
    if is_theano():
        return T.exp(a)
    else:
        return cgt.exp(a)

def prod(*args, **kwargs):
    import ipdb; ipdb.set_trace()

def sum(x, axis=None):
    if is_theano():
        return T.sum(x, axis=axis)
    else:
        return cgt.sum(x, axis=axis)

def dot(x, y):
    if is_theano():
        return T.dot(x, y)
    else:
        return cgt.dot(x, y)

def dimshuffle(x, *pattern):
    if is_theano():
        return x.dimshuffle(*pattern)
    else:
        return cgt.dimshuffle(x, list(pattern))

def tanh(x):
    if is_theano():
        return theano.tensor.tanh(x)
    else:
        return cgt.tanh(x)

def _ensure_broadcastable(a, b, bcpat):
    x, y = a, b
    xpat, ypat = bcpat.split(',')
    xpat = xpat.strip()
    ypat = ypat.strip()
    for i, xent in enumerate(xpat):
        if xent == '1' and not a.broadcastable[i]:
            raise ValueError('The %dth dimension of %s is not broadcastable', i, str(a))
    for i, yent in enumerate(ypat):
        if yent == '1' and not b.broadcastable[i]:
            raise ValueError('The %dth dimension of %s is not broadcastable', i, str(b))

def broadcast(x, a, b, bcpat):
    if is_theano():
        _ensure_broadcastable(a, b, bcpat)
        if x == '+':
            return a + b
        if x == '*':
            return a * b
        import ipdb; ipdb.set_trace()
    else:
        return cgt.broadcast(x, a, b, bcpat)

def reshape(x, shp):
    if is_theano():
        return T.reshape(x, shp)
    else:
        return cgt.reshape(x, shp)

def stack(tensors, axis=0):
    if is_theano():
        return T.stack(tensors, axis=axis)
    else:
        if axis is not 0:
            raise ValueError('only axis=0 is supported under cgt')
        return cgt.concatenate(map(lambda x: cgt.reshape(x, [1] + x.shape), tensors), axis=0)

def ones(shape):
    if is_theano():
        return T.ones(shape)
    else:
        return cgt.ones(shape)

def concatenate(items, axis=0):
    if is_theano():
        return T.concatenate(items, axis=axis)
    else:
        return cgt.concatenate(items, axis=axis)

def sqrt(x):
    if is_theano():
        return T.sqrt(x)
    else:
        return cgt.sqrt(x)
