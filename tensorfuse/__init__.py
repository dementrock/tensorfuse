# import gradient
# from .config import is_theano, is_cgt, is_tf, is_mxnet, floatX
# import compat
# import tensor
# from gradient import grad
# from utils import wrap_into_list
# import random
# import numpy as np
# from scan_op import scan
#
# if is_theano():
#     import theano
# elif is_cgt():
#     import cgt
# elif is_tf():
#     import tensorflow as tf
# elif is_mxnet():
#     import mxnet as mx
# else:
#     raise ValueError('Unknown backend')
#
#
# class TfFunctionWrapper(object):
#
#     def __init__(self, inputs, outputs, updates, givens):
#         self._inputs = inputs or []
#         self._outputs = outputs or []
#         self._updates = updates or {}
#         if givens:
#             raise NotImplementedError
#         self._givens = givens or {}
#
#         self._output_list = wrap_into_list(self._outputs)
#         if self._updates:
#             # cache graph construction
#             self._update_op = tf.group(
#                 *[tf.assign(var, val) for var, val in self._updates.iteritems()])
#         else:
#             self._update_op = None
#
#     def __call__(self, *args):
#         session = compat.tf_get_session()
#         compat.tf_ensure_init_variables()
#         try:
#             if self._update_op:
#                 output_vals = session.run(
#                     self._output_list + [self._update_op], feed_dict=dict(zip(self._inputs, args)))[:-1]
#             else:
#                 output_vals = session.run(
#                     self._output_list, feed_dict=dict(zip(self._inputs, args)))
#             if isinstance(self._outputs, (list, tuple)):
#                 return output_vals
#             else:
#                 return output_vals[0]
#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             import ipdb
#             ipdb.set_trace()
#
#
# class MxFunctionWrapper(object):
#
#     def __init__(self, inputs, outputs, updates, givens):
#         self._inputs = inputs or []
#         self._outputs = outputs or []
#         self._updates = updates or {}
#
#         if isinstance(outputs, (list, tuple)):
#             output_sym = mx.symbol.Group(outputs)
#         else:
#             output_sym = outputs
#         names = [x.list_outputs()[0] for x in inputs]
#         self._names = names
#         self._output_sym = output_sym
#         # expr = output_sym.bind(ctx=mx.cpu(), args=args)
#         # import ipdb; ipdb.set_trace()
#         if givens:
#             raise NotImplementedError
#         self._givens = givens or {}
#         self._executor = None
#
#     def __call__(self, *args):
#         vals = [mx.nd.array(x) for x in args]
#         if not self._executor:
#             executor = self._output_sym.bind(
#                 ctx=mx.cpu(), args=dict(zip(self._names, vals)))
#             self._executor = executor
#             self._executor.forward()
#         else:
#             self._executor.forward(**dict(zip(self._names, vals)))
#         return self._executor.outputs
#
#
# def function(inputs, outputs, updates=None, givens=None, allow_input_downcast=None, on_unused_input=None):
#     if is_theano():
#         allow_input_downcast = allow_input_downcast or False
#         on_unused_input = on_unused_input or 'raise'
#         return theano.function(inputs, outputs, updates=updates, givens=givens, allow_input_downcast=allow_input_downcast, on_unused_input=on_unused_input)
#     elif is_cgt():
#         return cgt.function(inputs, outputs, updates=updates, givens=givens)
#     elif is_tf():
#         return TfFunctionWrapper(inputs=inputs, outputs=outputs, updates=updates, givens=givens)
#     elif is_mxnet():
#         return MxFunctionWrapper(inputs=inputs, outputs=outputs, updates=updates, givens=givens)
#
#
# def shared(val, name=None, broadcastable=None, borrow=False):
#     if is_theano():
#         return theano.shared(val, name=name, broadcastable=broadcastable)
#     elif is_cgt():
#         return cgt.shared(val, name=name)
#     else:
#         var = tf.Variable(val.astype(floatX), name=name)
#         var._tensorfuse_shape_template = val.shape
#         var._tensorfuse_shared = True
#         compat.tf_add_blank_var(var)
#         return var
