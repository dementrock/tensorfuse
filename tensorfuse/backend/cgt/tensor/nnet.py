import cgt
import cgt.nn


def sigmoid(x):
    return cgt.sigmoid(x)


def relu(x):
    return cgt.nn.rectify(x)


def softmax(x):
    return cgt.nn.softmax(x)


def conv2d(input, filters, input_shape=None, filter_shape=None,
           border_mode='valid', subsample=(1, 1), filter_flip=True,
           image_shape=None, **kwargs):
    if border_mode == 'valid':
        padding = (0, 0)
    elif isinstance(border_mode, tuple):
        padding = border_mode
    else:
        raise NotImplementedError
    return cgt.nn.conv2d(
        x_BKRC=input,
        f_LKrc=filters,
        kernelshape=filters.shape[2:],
        pad=padding,
        stride=subsample,
    )
