from tensorfuse.config import is_theano, is_cgt, is_tf
if is_theano():
    from tensorfuse.backend.theano.gradient import *
elif is_cgt():
    from tensorfuse.backend.cgt.gradient import *
elif is_tf():
    from tensorfuse.backend.tensorflow.gradient import *
else:
    raise ValueError('Unknown backend')
