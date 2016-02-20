import cgt


def uniform(size=(), low=0.0, high=1.0, ndim=None):
    return cgt.rand(*size) * (high - low) + low


def normal(size=(), avg=0.0, std=1.0, ndim=None):
    return cgt.randn(*size) * std + avg


def binomial(size=(), n=1, p=0.5, ndim=None, dtype='int64'):
    return cgt.less(cgt.rand(*size), p)
