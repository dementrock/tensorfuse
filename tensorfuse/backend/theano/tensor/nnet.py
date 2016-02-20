import theano.tensor.nnet


def sigmoid(x):
    return theano.tensor.nnet.sigmoid(x)


def relu(x):
    return theano.tensor.nnet.relu(x)


def softmax(x):
    return theano.tensor.nnet.softmax(x)


def softplus(x):
    return theano.tensor.nnet.softplus(x)


def conv2d(input, filters, input_shape=None, filter_shape=None,
           border_mode='valid', subsample=(1, 1), filter_flip=True,
           image_shape=None, **kwargs):
    return theano.tensor.nnet.conv2d(
        input=input, filters=filters, input_shape=input_shape,
        filter_shape=filter_shape, border_mode=border_mode,
        subsample=subsample, filter_flip=filter_flip,
        image_shape=image_shape, **kwargs
    )


def categorical_crossentropy(coding_dist, true_dist):
    return theano.tensor.nnet.categorical_crossentropy(coding_dist, true_dist)
