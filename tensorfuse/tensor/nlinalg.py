from tensorfuse.config import is_theano, is_cgt, is_tf
if is_theano():
    from tensorfuse.backend.theano.tensor.nlinalg import *
elif is_cgt():
    from tensorfuse.backend.cgt.tensor.nlinalg import *
elif is_tf():
    from tensorfuse.backend.tensorflow.tensor.nlinalg import *
else:
    raise ValueError('Unknown backend')
