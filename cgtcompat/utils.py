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


