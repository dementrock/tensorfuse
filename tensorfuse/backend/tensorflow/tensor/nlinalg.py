import tensorflow as tf
from tensorfuse.compat import tf_var_from_shape


def matrix_inverse(x):
    return tf.matrix_inverse(x)
