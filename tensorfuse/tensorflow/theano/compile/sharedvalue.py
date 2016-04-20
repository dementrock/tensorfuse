from __future__ import print_function
from __future__ import absolute_import


class SharedVariableMeta(type):
    def __instancecheck__(cls, inst):
        return hasattr(inst, '_tensorfuse_shared')


class SharedVariable(object):
    __metaclass__ = SharedVariableMeta
