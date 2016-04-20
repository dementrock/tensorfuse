from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf


class VariableMeta(type):
    def __instancecheck__(cls, inst):
        return isinstance(inst, (tf.Tensor, tf.Variable))


class Variable(object):
    __metaclass__ = VariableMeta


# modified from cgt's topsorted code
def _tf_topsorted(outputs):
    assert isinstance(outputs, (list, tuple))
    marks = {}
    out = []
    stack = []  # pylint: disable=W0621
    # i: node
    # jidx = number of children visited so far from that node
    # marks: state of each node, which is one of
    #   0: haven't visited
    #   1: have visited, but not done visiting children
    #   2: done visiting children
    for x in outputs:
        stack.append((x, 0))
        while stack:
            (i, jidx) = stack.pop()
            if jidx == 0:
                m = marks.get(i, 0)
                if m == 0:
                    marks[i] = 1
                elif m == 1:
                    raise ValueError("not a dag")
                else:
                    continue
            ps = list(i.op.inputs)
            if jidx == len(ps):
                marks[i] = 2
                out.append(i)
            else:
                stack.append((i, jidx + 1))
                j = ps[jidx]
                stack.append((j, 0))
    return out


def _tf_is_input(x):
    return len(x.op.inputs) == 0


def inputs(outputs):
    outputs = list(outputs)
    return [node for node in _tf_topsorted(outputs) if _tf_is_input(node)]
