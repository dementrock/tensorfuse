from theano.tensor.shared_randomstreams import RandomStreams
_srng = RandomStreams(seed=234)


def uniform(size=(), low=0.0, high=1.0, ndim=None):
    return _srng.uniform(size=size, low=low, high=high, ndim=ndim)


def normal(size=(), avg=0.0, std=1.0, ndim=None):
    return _srng.normal(size=size, avg=avg, std=std, ndim=ndim)


def binomial(size=(), n=1, p=0.5, ndim=None, dtype='int64'):
    return _srng.binomial(size=size, n=n, p=p, ndim=ndim, dtype=dtype)
