from .abstract import Node
from abc import ABCMeta, abstractmethod


class Variable(Node, metaclass=ABCMeta):
    def __init__(self, value):
        self.value = value

__all__ = [
    'Variable'
]