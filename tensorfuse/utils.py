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


# From Theano
def format_as(use_list, use_tuple, outputs):
    """
    Formats the outputs according to the flags `use_list` and `use_tuple`.
    If `use_list` is True, `outputs` is returned as a list (if `outputs`
    is not a list or a tuple then it is converted in a one element list).
    If `use_tuple` is True, `outputs` is returned as a tuple (if `outputs`
    is not a list or a tuple then it is converted into a one element tuple).
    Otherwise (if both flags are false), `outputs` is returned.
    """
    assert not (use_list and use_tuple), \
        "Both flags cannot be simultaneously True"
    if (use_list or use_tuple) and not isinstance(outputs, (list, tuple)):
        if use_list:
            return [outputs]
        else:
            return (outputs,)
    elif not (use_list or use_tuple) and isinstance(outputs, (list, tuple)):
        assert len(outputs) == 1, \
            "Wrong arguments. Expected a one element list"
        return outputs[0]
    elif use_list or use_tuple:
        if use_list:
            return list(outputs)
        else:
            return tuple(outputs)
    else:
        return outputs
