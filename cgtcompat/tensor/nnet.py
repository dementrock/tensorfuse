from ..config import is_theano, is_cgt
if is_theano():
    import theano.tensor.nnet
else:
    import cgt

def sigmoid(x):
    if is_theano():
        return theano.tensor.nnet.sigmoid(x)
    else:
        return cgt.sigmoid(x)

def relu(x):
    if is_theano():
        return theano.tensor.nnet.relu(x)
    else:
        return cgt.nn.rectify(x)
