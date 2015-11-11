from .config import is_theano, is_cgt, is_tf, floatX
import numpy as np

if is_tf():
    import tensorflow as tf

def wrap_into_list(x):
    """
    Wrap the input into a list if it is not already a list.

    """
    if x is None:
        return []
    elif not isinstance(x, (list, tuple)):
        return [x]
    else:
        return list(x)

def wrap_into_tuple(x):
    """
    Wrap the input into a list if it is not already a list.

    """
    if x is None:
        return []
    elif not isinstance(x, (list, tuple)):
        return (x,)
    else:
        return tuple(x)
