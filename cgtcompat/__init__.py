import gradient
from .config import is_theano, is_cgt, is_tf, session
import compat
from operator import itemgetter
import tensor
from gradient import grad
from utils import wrap_into_list
import random

if is_theano():
    import theano
elif is_cgt():
    import cgt
else:
    import tensorflow as tf

class TfFunctionWrapper(object):

    def __init__(self, inputs, outputs, updates, givens):
        self._inputs = inputs or []
        self._outputs = outputs or []
        self._updates = updates or {}
        self._givens = givens or {}

    def __call__(self, *args):
        try:
            #tasks = []
            #for var, val in zip(self._inputs, args):
            #    tasks.append(tf.assign(var, val))
            compat.tf_ensure_init_variables()
            session.run([tf.assign(var, val) for var, val in zip(self._inputs, args)])#tasks)
            output_list = wrap_into_list(self._outputs)
            n_outputs = len(output_list)
            results = session.run(output_list + self._updates.values())
            output_vals = results[:n_outputs]
            update_vals = results[n_outputs:]
            session.run([tf.assign(var, val) for var, val in zip(self._updates.keys(), update_vals)])
            if isinstance(self._outputs, list):
                return output_vals
            else:
                return output_vals[0]
        except Exception as e:
            import ipdb; ipdb.set_trace()
        

def function(inputs, outputs, updates=None, givens=None, allow_input_downcast=None, on_unused_input=None):
    if is_theano():
        allow_input_downcast = allow_input_downcast or False
        on_unused_input = on_unused_input or 'raise'
        return theano.function(inputs, outputs, updates=updates, givens=givens, allow_input_downcast=allow_input_downcast, on_unused_input=on_unused_input)
    elif is_cgt():
        #if allow_input_downcast is not None:
        #    print 'allow_input_downcast ignored'
        #if on_unused_input is not None:
        #    print 'allow_input_downcast ignored'
        return cgt.function(inputs, outputs, updates=updates, givens=givens)
    else:
        return TfFunctionWrapper(inputs=inputs, outputs=outputs, updates=updates, givens=givens)

def shared(val, name=None, broadcastable=None, borrow=False):
    if is_theano():
        return theano.shared(val, name=name, broadcastable=broadcastable)
    elif is_cgt():
        return cgt.shared(val, name=name)
    else:
        var = tf.Variable(val, name=name)
        var._cgtcompat_shared = True
        var._cgtcompat_initialized = False
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
    if is_theano():
        return theano.scan(fn=fn, sequences=sequences, outputs_info=outputs_info, non_sequences=non_sequences, n_steps=n_steps, truncate_gradient=truncate_gradient, go_backwards=go_backwards, mode=mode, name=name, profile=profile, allow_gc=allow_gc, strict=strict)
    elif is_cgt() or is_tf():
        # n_steps must be provided under cgt or tensorflow
        if n_steps is None:
            raise ValueError('n_steps must be provided for scan to work under CGT / TensorFlow')
        sequences = wrap_into_list(sequences)
        non_sequences = wrap_into_list(non_sequences)
        output_as_list = isinstance(outputs_info, list)
        outputs_info = wrap_into_list(outputs_info)
        if go_backwards and n_steps < 0:
            go_backwards = False
            n_steps = -n_steps
        if go_backwards or n_steps < 0:
            go_backwards = True
            n_steps = abs(n_steps)
        if mode is not None: 
            import ipdb; ipdb.set_trace()
        if name is not None:
            print 'name ignored under cgt.scan'
        step_outputs = []
        cur_output = outputs_info
        loop_range = range(n_steps-1, -1, -1) if go_backwards else range(n_steps)
        for i in loop_range:
            cur_output = fn(*(map(itemgetter(i), sequences) + cur_output + non_sequences))
            step_outputs.append(cur_output)
        outputs = []
        if len(step_outputs) > 0:
            for i in range(len(outputs_info)):
                outputs.append(tensor.stack(map(itemgetter(i), step_outputs)))
        else:
            import ipdb; ipdb.set_trace()
        # This is quite ugly, but unfortunately it's what theano does
        if len(outputs) > 1:
            # update is not supported yet
            return outputs, None
        elif len(outputs) == 1:
            return outputs[0], None
        else:
            return None, None
    else:
        import ipdb; ipdb.set_trace()
