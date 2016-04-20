# import os
#
# THEANO = 0
# CGT = 1
# TENSORFLOW = 2
# TF = 2
# MXNET = 3
#
# mode = THEANO
#
# _session = None
#
#
# if 'TENSORFUSE_MODE' in os.environ:
#     if os.environ['TENSORFUSE_MODE'] == 'theano':
#         mode = THEANO
#     elif os.environ['TENSORFUSE_MODE'] == 'cgt':
#         mode = CGT
#     elif os.environ['TENSORFUSE_MODE'] in ['tensorflow', 'tf']:
#         mode = TENSORFLOW
#     elif os.environ['TENSORFUSE_MODE'] == 'mxnet':
#         mode = MXNET
#     else:
#         raise ValueError("Unrecognized environment variable TENSORFUSE %s: must be one of 'theano', 'cgt', 'mxnet', 'tensorflow', or 'tf'" % os.environ['TENSORFUSE_MODE'])
#
#
# def is_theano():
#     return mode == THEANO
#
#
# def is_cgt():
#     return mode == CGT
#
#
# def is_tensorflow():
#     return mode == TENSORFLOW
#
# is_tf = is_tensorflow
#
#
# def is_mxnet():
#     return mode == MXNET
#
#
# if is_theano():
#     print 'Using Theano for TensorFuse'
#     import theano
#     floatX = theano.config.floatX
# elif is_cgt():
#     print 'Using CGT for TensorFuse'
#     import cgt
#     floatX = cgt.floatX
# elif is_mxnet():
#     print 'Using mxnet for TensorFuse'
#     floatX = 'float32'
# else:
#     print 'Using TensorFlow for TensorFuse'
#     floatX = 'float32'
