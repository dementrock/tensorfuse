import cgt


def max_pool_2d(input, ds, ignore_border=None, st=None, padding=(0, 0),
                mode='max'):
    if not ignore_border:
        raise NotImplementedError
    if mode != 'max':
        raise NotImplementedError
    if st is None:
        st = ds
    return cgt.nn.max_pool_2d(input, kernelshape=ds, pad=padding, stride=st)
