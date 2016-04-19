import tensorfuse.tensorflow.theano as tf_theano
import tensorfuse.tensorflow.theano.tensor as tf_TT
import theano
import theano.tensor as TT
import numpy as np


def test_scan1():
    def result(theano, TT):
        def fn(s1, s2):
            return s1 + s2

        outputs, _ = theano.scan(
            fn,
            sequences=[TT.ones(10), 2 * TT.ones(10)])
        return theano.function([], outputs)()

    np.testing.assert_array_equal(result(theano, TT), result(tf_theano, tf_TT))


def test_scan2():
    def result(theano, TT):
        def fn(s1, s2, addn):
            return s1 + s2 + addn

        outputs, _ = theano.scan(
            fn,
            sequences=[TT.ones(10), 2 * TT.ones(10)],
            non_sequences=1,
        )
        return theano.function([], outputs)()

    np.testing.assert_array_equal(result(theano, TT), result(tf_theano, tf_TT))


def test_scan3():
    def result(theano, TT):
        def fn(s1, s2, o1):
            return s1 + s2 + o1

        outputs, _ = theano.scan(
            fn,
            sequences=[TT.ones(10), 2 * TT.ones(10)],
            outputs_info=0.,
        )
        return theano.function([], outputs)()

    np.testing.assert_array_equal(result(theano, TT), result(tf_theano, tf_TT))
