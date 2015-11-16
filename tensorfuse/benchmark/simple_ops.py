import tensorfuse as theano
from tensorfuse.config import is_tf
if is_tf():
    import tensorflow as tf
    from tensorfuse.compat import tf_reset_session
import tensorfuse.tensor as TT
import numpy as np
import time

def timeit(n):
    def wrap(f):
        if is_tf():
            gdef = tf.Graph().as_default()
            gdef.__enter__()
            tf_reset_session()
        executor = f()
        ts = time.time()
        for _ in range(n):
            executor()
        te = time.time()
        print 'func:%r %d times took: %2.4f sec' % \
          (f.__name__, n, te-ts)
        if is_tf():
            gdef.__exit__(None, None, None)
    return wrap


@timeit(10000)
def time_sin():
    z = TT.vector('z')
    f_sin = theano.function([z], TT.sin(z))
    def run():
        f_sin([1,2,3])
    return run

for mat_conf in [(3, 10000),
                 (64, 10000),
                 (256, 1000),
                 (512, 1000),
                 (1024, 100),
                 (2048, 10),
                 (4096, 10),
                 (8192, 10)]:
    def time_matmul():
        a = TT.matrix('a')
        b = TT.matrix('b')
        f_dot = theano.function([a, b], TT.dot(a, b), allow_input_downcast=True)
        a_val = np.random.rand(mat_conf[0], mat_conf[0]).astype('float32')
        b_val = np.random.rand(mat_conf[0], mat_conf[0]).astype('float32')
        def run():
            f_dot(a_val, b_val)
        return run
    time_matmul.__name__ = 'time_matmul_%dx%d' % (mat_conf[0], mat_conf[0])
    timeit(mat_conf[1])(time_matmul)

@timeit(10000)
def time_slicing():
    z = TT.vector('z')
    f_slice = theano.function([z], [z[:2], z[:]])
    def run():
        f_slice([1,2,3])
    return run

@timeit(10000)
def time_noop():
    # This is needed to make TensorFlow happy
    TT.constant(0)
    return theano.function([], [])
