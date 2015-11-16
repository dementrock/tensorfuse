from utils import wrap_into_list
import tensorflow as tf


class TfFunctionWrapper(object):

    def __init__(self, inputs, outputs, updates, givens):
        self._inputs = inputs or []
        self._outputs = outputs or []
        self._updates = updates or {}
        self._givens = givens or {}

        self._output_list = wrap_into_list(self._outputs)
        if self._updates:
            # cache graph construction
            self._update_op = tf.group(*[tf.assign(var, val) for var, val in self._updates.iteritems()])
        else:
            self._update_op = None

    def __call__(self, *args):
        session = compat.tf_get_session()
        compat.tf_ensure_init_variables()
        try:
            if self._update_op:
                output_vals = session.run(self._output_list + [self._update_op], feed_dict=dict(zip(self._inputs, args)))[:-1]
            else:
                output_vals = session.run(self._output_list, feed_dict=dict(zip(self._inputs, args)))
            if isinstance(self._outputs, (list, tuple)):
                return output_vals
            else:
                return output_vals[0]
        except Exception as e:
            import ipdb; ipdb.set_trace()


def function(inputs, outputs, updates=None, givens=None, allow_input_downcast=None, on_unused_input=None):
    return TfFunctionWrapper(inputs=inputs, outputs=outputs, updates=updates, givens=givens)


def shared(val, name=None, broadcastable=None, borrow=False):
    var = tf.Variable(val.astype(floatX), name=name)
    var._tensorfuse_shape_template = val.shape
    var._tensorfuse_shared = True
    compat.tf_add_blank_var(var)
    return var
