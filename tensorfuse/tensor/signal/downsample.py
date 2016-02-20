from tensorfuse.config import is_theano, is_cgt, is_tf, is_mxnet
if is_theano():
    from tensorfuse.backend.theano.tensor.signal.downsample import *
elif is_cgt():
    from tensorfuse.backend.cgt.tensor.signal.downsample import *
elif is_tf():
    from tensorfuse.backend.tensorflow.tensor.signal.downsample import *
elif is_mxnet():
    from tensorfuse.backend.mxnet.tensor.signal.downsample import *
else:
    raise ValueError('Unknown backend')
