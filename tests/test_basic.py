import tensorfuse.tensorflow.theano as tf_theano
import tensorfuse.tensorflow.theano.tensor as tf_TT
import theano
import theano.tensor as TT
import numpy as np


def test_compile():
    def result(theano, TT):
        x = TT.vector('x')
        return theano.function([x], x ** 2)([1, 2])

    np.testing.assert_array_equal(result(theano, TT), result(tf_theano, tf_TT))


def test_membership():
    def result(theano, TT):
        x = TT.vector('x')
        return isinstance(x, TT.Variable)

    assert result(theano, TT) == result(tf_theano, tf_TT)


def test_grad_clip():
    def result(theano, TT):
        x = TT.vector('x')
        x_clipped = theano.gradient.grad_clip(x, -1, 1)
        return (theano.function([x], TT.grad(TT.sum(x_clipped * x_clipped), x))([100, 100]))

    np.testing.assert_array_equal(result(theano, TT), result(tf_theano, tf_TT))


def test_switch():
    def result(theano, TT):
        x = TT.matrix('x')
        y = TT.vector('y')
        yshift = y.dimshuffle(0, 'x')
        z = TT.switch(yshift, x, 1 - x)
        f_z = theano.function([x, y], z)
        result = f_z(np.ones((10, 10), dtype=theano.config.floatX), np.ones((10,), dtype=theano.config.floatX))
        return result

    np.testing.assert_array_equal(result(theano, TT), result(tf_theano, tf_TT))

# TODO test reshape ([1, 1])
