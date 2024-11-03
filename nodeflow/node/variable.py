from .abstract import Node
from .function import Function
from abc import ABCMeta
from typing import Any


class Variable(Node, metaclass=ABCMeta):
    def __init__(self, value: Any):
        self.value = value

    def __rshift__(self, other: Function) -> 'Variable':
        return other.compute(self)

__all__ = [
    'Variable'
]