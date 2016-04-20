import tensorflow as tf


def solve(a, b):
    if b.ndim == 1:
        return tf.reshape(tf.matmul(tf.matrix_inverse(a), tf.expand_dims(b, -1)), [-1])
    elif b.ndim == 2:
        return tf.matmul(tf.matrix_inverse(a), b)
    else:
        import ipdb; ipdb.set_trace()
