from tensorfuse.config import is_theano, is_cgt, is_tf
if is_theano():
    import theano
    import theano.tensor as T
    import theano.tensor.nlinalg
elif is_cgt():
    import cgt
else:
    import tensorflow as tf
    from tensorfuse.compat import tf_var_from_shape

def matrix_inverse(x):
    if is_theano():
        return theano.tensor.nlinalg.matrix_inverse(x)
    elif is_cgt():
        import ipdb; ipdb.set_trace()
    elif is_tf():
        return tf.matrix_inverse(x)
    else:
        import ipdb; ipdb.set_trace()

