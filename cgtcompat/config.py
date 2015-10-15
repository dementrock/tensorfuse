THEANO = 0
CGT = 1
mode = CGT#THEANO#CGT#THEANO

def is_theano():
    return mode == THEANO

def is_cgt():
    return mode == CGT

if is_theano():
    import theano
    floatX = theano.config.floatX
else:
    import cgt
    floatX = cgt.floatX
