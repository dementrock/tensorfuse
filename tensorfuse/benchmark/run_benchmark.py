import sys
import os

if len(sys.argv) != 2:
    print 'Usage: python run_benchmark.py FILE'

for env in ['mxnet', 'theano', 'cgt', 'tensorflow']:
    os.system('TENSORFUSE_MODE=%s python %s' % (env, sys.argv[1]))
