from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf


def grad_clip(x, lower_bound, upper_bound):
    return x


def grad(cost, wrt, known_grads=None, disconnected_inputs=None):
    ret = tf.gradients(cost, wrt)
    if isinstance(wrt, list):
        return [x if x is not None else tf.zeros_like(wrt[i]) for i, x in enumerate(ret)]
    elif ret[0] is not None:
        return ret[0]
    else:
        return tf.zeros_like(wrt)
