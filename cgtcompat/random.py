from .config import is_theano
if is_theano():
    import theano
    from theano.tensor.shared_randomstreams import RandomStreams
    _srng = RandomStreams(seed=234)
else:
    import cgt

def uniform(size=(), low=0.0, high=1.0, ndim=None):
    if is_theano():
        return _srng.uniform(size=size, low=low, high=high, ndim=ndim)
    else:
        import ipdb; ipdb.set_trace()
