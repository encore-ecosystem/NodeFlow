from abc import ABCMeta, abstractmethod
from nodeflow.node.abstract import Node
from nodeflow.node.variable import Variable


class Function(Node, metaclass=ABCMeta):
    @abstractmethod
    def compute(self, *args, **kwargs) -> Variable:
        raise NotImplementedError


__all__ = [
    'Function'
]