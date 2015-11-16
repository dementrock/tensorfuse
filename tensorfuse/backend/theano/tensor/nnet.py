import theano.tensor.nnet


def sigmoid(x):
    return theano.tensor.nnet.sigmoid(x)


def relu(x):
    return theano.tensor.nnet.relu(x)


def softmax(x):
    return theano.tensor.nnet.softmax(x)
