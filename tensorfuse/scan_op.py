from .config import is_theano, is_cgt, is_tf, floatX
if is_theano():
    import theano
elif is_cgt():
    import cgt
else:
    import tensorflow as tf
from utils import wrap_into_list
from operator import itemgetter


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
        if outputs_info is not None:
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
            # Only pass output if needed
            if outputs_info is not None:
                cur_output = fn(*(map(itemgetter(i), sequences) + cur_output + non_sequences))
            else:
                cur_output = fn(*(map(itemgetter(i), sequences) + non_sequences))
            step_outputs.append(cur_output)
        outputs = []
        try:
            if len(step_outputs) > 0:
                if outputs_info is None:
                    for i in range(len(step_outputs[0])):
                        outputs.append(tf.pack(map(itemgetter(i), step_outputs)))
                    #outputs = step_outputs
                else:
                    for i in range(len(outputs_info)):
                        outputs.append(tf.pack(map(itemgetter(i), step_outputs)))
            else:
                import ipdb; ipdb.set_trace()
        except Exception as e:
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
