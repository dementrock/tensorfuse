from tensorfuse.config import is_theano, is_cgt, is_tf
if is_theano():
    import theano
    import theano.tensor as T
    import theano.tensor.slinalg
elif is_cgt():
    import cgt
else:
    import tensorflow as tf

def solve(a, b):
    if is_theano():
        return theano.tensor.slinalg.solve(a, b)
    elif is_cgt():
        import ipdb; ipdb.set_trace()
    elif is_tf():
        if b.ndim == 1:
            return tf.reshape(tf.matmul(tf.matrix_inverse(a), tf.expand_dims(b, -1)), [-1])
        elif b.ndim == 2:
            return tf.matmul(tf.matrix_inverse(a), b)
        else:
            import ipdb; ipdb.set_trace()
    else:
        import ipdb; ipdb.set_trace()
