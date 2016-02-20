from tensorfuse.config import is_theano, is_cgt, is_tf, is_mxnet
if is_theano():
    from tensorfuse.backend.theano.random import *
elif is_cgt():
    from tensorfuse.backend.cgt.random import *
elif is_tf():
    from tensorfuse.backend.tensorflow.random import *
elif is_mxnet():
    from tensorfuse.backend.mxnet.random import *
else:
    raise ValueError('Unknown backend')
