import tensorflow as tf


def conv2d(input, filters, input_shape=None, filter_shape=None,
           border_mode='valid', subsample=(1, 1), filter_flip=True,
           image_shape=None, **kwargs):
    if border_mode == 'valid' or border_mode == (0, 0):
        padding = 'VALID'
    elif border_mode == 'same':
        padding = 'SAME'
    else:
        raise NotImplementedError
    if filter_flip:
        # reverse the filter before applying it
        filters = filters[:, :, ::-1, ::-1]
    reshaped_input = tf.transpose(input, perm=[0, 2, 3, 1])
    reshaped_filters = tf.transpose(filters, perm=[2, 3, 1, 0])
    return tf.transpose(
        tf.nn.conv2d(
            input=reshaped_input,
            filter=reshaped_filters,
            strides=[1, subsample[0], subsample[1], 1],
            padding=padding
        ), perm=[0, 3, 1, 2]
    )


def relu(x):
    return tf.nn.relu(x)


def softmax(x):
    return tf.nn.softmax(x)


def categorical_crossentropy(coding_dist, true_dist):
    if true_dist.ndim == coding_dist.ndim:
        return -tf.reduce_sum(
            tf.cast(true_dist, 'float32') * tf.log(coding_dist),
            reduction_indices=[coding_dist.ndim - 1]
        )
    elif true_dist.ndim == coding_dist.ndim - 1:
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.log(coding_dist), tf.cast(true_dist, 'int64')
        )
    else:
        raise TypeError('rank mismatch between coding and true distributions')
