from theano.tensor.signal import downsample


def max_pool_2d(input, ds, ignore_border=None, st=None, padding=(0, 0),
                mode='max'):
    return downsample.max_pool_2d(input, ds=ds, ignore_border=ignore_border,
                                  st=st, padding=padding, mode=mode)
