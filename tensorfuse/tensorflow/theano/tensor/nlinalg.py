import tensorflow as tf

import theano
theano.scan


def matrix_inverse(x):
    return tf.matrix_inverse(x)
