import os

THEANO = 0
CGT = 1
mode = CGT

if 'CGT_COMPAT_MODE' in os.environ:
    if os.environ['CGT_COMPAT_MODE'] == 'theano':
        mode = THEANO
    elif os.environ['CGT_COMPAT_MODE'] == 'cgt':
        mode = CGT
    else:
        raise ValueError('Unrecognized environment variable CGT_COMPAT_MODE %s: must be either theano or cgt' % os.environ['CGT_COMPAT_MODE'])

def is_theano():
    return mode == THEANO

def is_cgt():
    return mode == CGT

if is_theano():
    print 'Using Theano for CGT compatibility mode'
    import theano
    floatX = theano.config.floatX
else:
    print 'Using CGT for CGT compatibility mode'
    import cgt
    floatX = cgt.floatX
