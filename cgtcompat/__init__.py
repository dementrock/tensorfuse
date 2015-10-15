import gradient
from .config import is_theano, is_cgt
import compat

if is_theano():
    import theano
else:
    import cgt

def function(inputs, outputs, updates=None, givens=None, allow_input_downcast=None, on_unused_input=None):
    if is_theano():
        allow_input_downcast = allow_input_downcast or False
        on_unused_input = on_unused_input or 'raise'
        return theano.function(inputs, outputs, updates=updates, givens=givens, allow_input_downcast=allow_input_downcast, on_unused_input=on_unused_input)
    else:
        #if allow_input_downcast is not None:
        #    print 'allow_input_downcast ignored'
        #if on_unused_input is not None:
        #    print 'allow_input_downcast ignored'
        return cgt.function(inputs, outputs, updates=updates, givens=givens)

def shared(val, name=None):#*args, **kwargs):
    if is_theano():
        return theano.shared(val, name=name)#*args, **kwargs)
    else:
        return cgt.shared(val, name=name)
