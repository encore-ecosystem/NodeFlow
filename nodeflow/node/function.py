from .abstract import Node
from abc import ABCMeta, abstractmethod


class Function(Node, metaclass=ABCMeta):
    @abstractmethod
    def compute(self, *args, **kwargs) -> Node:
        raise NotImplementedError


__all__ = [
    'Function'
]