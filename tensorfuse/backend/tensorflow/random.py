import tensorflow as tf


def uniform(size=(), low=0.0, high=1.0, ndim=None):
    return tf.random_uniform(size, low, high)


def normal(size=(), avg=0.0, std=1.0, ndim=None):
    return tf.random_normal(shape=size, mean=avg, stddev=std)


def binomial(size=(), p=0.5, ndim=None, dtype='int64'):
    return tf.cast(tf.less(tf.random_uniform(size, 0., 1.), p), dtype)
