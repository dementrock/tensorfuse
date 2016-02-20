import tensorflow as tf


def max_pool_2d(input, ds, ignore_border=None, st=None, padding=(0, 0),
                mode='max'):
    reshaped_input = tf.transpose(input, perm=[0, 2, 3, 1])
    if st is None:
        st = ds
    if padding == (0, 0):
        padding = 'VALID'
    else:
        raise NotImplementedError
    if mode == 'max':
        ret = tf.nn.max_pool(
            value=reshaped_input,
            ksize=[1, ds[0], ds[1], 1],
            strides=[1, st[0], st[1], 1],
            padding=padding,
        )
    # tweat both as the same for now
    elif mode == 'average_inc_pad' or mode == 'average_exc_pad':
        ret = tf.nn.avg_pool(
            value=reshaped_input,
            ksize=[1, ds[0], ds[1], 1],
            strides=[1, st[0], st[1], 1],
            padding=padding,
        )
    else:
        raise NotImplementedError
    return tf.transpose(ret, perm=[0, 3, 1, 2])
