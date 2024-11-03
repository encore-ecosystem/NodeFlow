from .abstract import Node
from abc import ABCMeta
from typing import Any

class Variable(Node, metaclass=ABCMeta):
    def __init__(self, value: Any):
        self.value = value
    #                               (1)
    def __rshift__(self, other: 'Function') -> 'Variable':
        return other.compute(self)


# The import leaves here for resolving circular import. Also see (1) above line 8
from nodeflow.node.function import Function

__all__ = [
    'Variable'
]