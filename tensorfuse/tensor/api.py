from tensorfuse.config import is_theano, is_cgt, is_tf, is_mxnet
if is_theano():
    from tensorfuse.backend.theano.tensor.api import *
elif is_cgt():
    from tensorfuse.backend.cgt.tensor.api import *
elif is_tf():
    from tensorfuse.backend.tensorflow.tensor.api import *
elif is_mxnet():
    from tensorfuse.backend.mxnet.tensor.api import *
else:
    raise ValueError('Unknown backend')
