import cgt
import cgt.nn


def sigmoid(x):
    return cgt.sigmoid(x)


def relu(x):
    return cgt.nn.rectify(x)


def softmax(x):
    return cgt.nn.softmax(x)
