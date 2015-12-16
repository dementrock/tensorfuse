from tensorfuse.config import is_theano, is_cgt, is_tf
if is_theano():
    import theano
    from theano.tensor.shared_randomstreams import RandomStreams
    _srng = RandomStreams(seed=234)
elif is_cgt():
    import cgt
elif is_tf():
    import tensorflow as tf


def uniform(size=(), low=0.0, high=1.0, ndim=None):
    if is_theano():
        return _srng.uniform(size=size, low=low, high=high, ndim=ndim)
    elif is_cgt():
        import ipdb; ipdb.set_trace()
    elif is_tf():
        return tf.random_uniform(size, low, high)
    else:
        import ipdb; ipdb.set_trace()
