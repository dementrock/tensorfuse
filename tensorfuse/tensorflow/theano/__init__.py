from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorfuse.utils import wrap_into_list
from tensorfuse.tensorflow import compat
from . import config
from . import compile
from . import gradient
from . import tensor
from .gradient import grad
from .gof import Variable
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.framework import ops
from tensorfuse.tensorflow.compat import get_raw_dimensions


class TfFunctionWrapper(object):
    def __init__(self, inputs, outputs, updates, givens):
        if inputs is not None:
            self._inputs = inputs
        else:
            self._inputs = []
        if outputs is not None:
            self._outputs = outputs
        else:
            self._outputs = []
        if updates is not None:
            self._updates = updates
        else:
            self._updates = {}
        if givens:
            raise NotImplementedError
        self._givens = givens or {}

        self._output_list = wrap_into_list(self._outputs)
        if self._updates:
            # cache graph construction
            self._update_op = tf.group(
                *[tf.assign(var, val) for var, val in self._updates.iteritems()])
        else:
            self._update_op = None

    def __call__(self, *args):
        session = compat.tf_get_session()
        compat.tf_ensure_init_variables()
        try:
            if self._update_op:
                output_vals = session.run(
                    self._output_list + [self._update_op], feed_dict=dict(zip(self._inputs, args)))[:-1]
            else:
                output_vals = session.run(
                    self._output_list, feed_dict=dict(zip(self._inputs, args)))
            if isinstance(self._outputs, (list, tuple)):
                return output_vals
            else:
                return output_vals[0]
        except Exception as e:
            import traceback
            traceback.print_exc()
            import ipdb
            ipdb.set_trace()


def function(inputs, outputs, updates=None, givens=None, allow_input_downcast=None, on_unused_input=None, mode=None):
    return TfFunctionWrapper(inputs=inputs, outputs=outputs, updates=updates, givens=givens)


def shared(val, name=None, broadcastable=None, borrow=False):
    var = tf.Variable(val.astype(config.floatX), name="x")
    var._tensorfuse_shape_template = val.shape
    var._tensorfuse_shared = True
    compat.tf_add_blank_var(var)
    return var


def scan(fn,
         sequences=None,
         outputs_info=None,
         non_sequences=None,
         n_steps=None,
         truncate_gradient=-1,
         go_backwards=False,
         mode=None,
         name=None,
         profile=False,
         allow_gc=None,
         strict=False):
    sequences = wrap_into_list(sequences)
    non_sequences = wrap_into_list(non_sequences)
    for x in sequences:
        if isinstance(x, dict):
            raise NotImplementedError
    for x in non_sequences:
        if isinstance(x, dict):
            raise NotImplementedError
    if outputs_info is not None:
        outputs_info = wrap_into_list(outputs_info)

    # if go_backwards:
    #     raise NotImplementedError


    if n_steps is not None and n_steps < 0:
        raise NotImplementedError
    # if go_backwards and n_steps < 0:
    #     go_backwards = False
    #     n_steps = -n_steps
    # if go_backwards or n_steps < 0:
    #     go_backwards = True
    #     n_steps = abs(n_steps)
    if mode is not None:
        raise NotImplementedError
    if name is not None:
        raise NotImplementedError
    if truncate_gradient != -1:
        raise NotImplementedError
    # step_outputs = []
    # cur_output = outputs_info
    # loop_range = range(n_steps - 1, -1, -1) if go_backwards else range(n_steps)

    if not callable(fn):
        raise TypeError("fn must be callable.")

    with variable_scope.variable_op_scope(sequences + non_sequences, name, "scan") as varscope:
        # Any get_variable calls fn will cache the first call locally
        # and not issue repeated network I/O requests for each iteration.
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # Convert sequences to tensor array.
        n = array_ops.shape(sequences[0])[0]

        sequences_ta = [tensor_array_ops.TensorArray(dtype=e.dtype, size=n, dynamic_size=False) for e in sequences]
        sequences_ta = [e_ta.unpack(e) for e_ta, e in zip(sequences_ta, sequences)]

        # run the computation graph once to get the output types
        if outputs_info is None:
            outputs_data = wrap_into_list(fn(*([x[0] for x in sequences] + non_sequences)))
        else:
            outputs_data = [ops.convert_to_tensor(x) for x in outputs_info]

        acc_ta = [tensor_array_ops.TensorArray(dtype=a.dtype, size=n, dynamic_size=False) for a in outputs_data]

        i = constant_op.constant(0)

        if outputs_info is None:
            fn_shapes = map(compat.get_raw_dimensions, [x[0] for x in sequences] + non_sequences)
        else:
            fn_shapes = map(compat.get_raw_dimensions, [x[0] for x in sequences] + outputs_info + non_sequences)
        fn_shapes = map(list, fn_shapes)

        def compute(i, *all_inputs):
            if go_backwards:
                fn_inputs = [e_ta.read(n - 1 - i) for e_ta in sequences_ta]
            else:
                fn_inputs = [e_ta.read(i) for e_ta in sequences_ta]
            acc_ta_n = list(all_inputs[:len(acc_ta)])
            prev_n = list(all_inputs[len(acc_ta):])
            if outputs_info is not None:
                fn_inputs.extend(prev_n)
            fn_inputs.extend(non_sequences)
            for fn_input, shape in zip(fn_inputs, fn_shapes):
                if isinstance(fn_input, tf.Tensor):
                    fn_input.set_shape(shape)
            a_n = wrap_into_list(fn(*fn_inputs))
            ta_n = [ta.write(i, a) for ta, a in zip(acc_ta_n, a_n)]
            if outputs_info is not None:
                return [i + 1] + ta_n + a_n
            else:
                return [i + 1] + ta_n

        if outputs_info is None:
            initial_tensors = [i] + acc_ta
        else:
            initial_tensors = [i] + acc_ta + outputs_data
        results = control_flow_ops.while_loop(
            (lambda i, *_: i < n), compute, initial_tensors
        )[1:]

        outputs = [r_a.pack() for r_a in results[:len(acc_ta)]]

        for o, o_tmpl in zip(outputs, outputs_data):
            if isinstance(o_tmpl, tf.Variable):
                shape = list(o_tmpl._initial_value._shape)
            else:
                shape = list(o_tmpl._shape)
            o.set_shape([tf.Dimension(None)] + shape)

        # This is quite ugly, but unfortunately it's what Theano does
        # Update is not supported yet
        if len(outputs) > 1:
            return outputs, None
        elif len(outputs) == 1:
            return outputs[0], None
        else:
            return None, None
