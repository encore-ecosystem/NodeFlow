from nodeflow.node.abstract import Node
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Type
import inspect


class Function(Node, metaclass=ABCMeta):
    @abstractmethod
    #                                         (1)
    def compute(self, *args, **kwargs) -> 'Variable':
        raise NotImplementedError

    def get_parameters(self) -> OrderedDict[str, Type['Variable']]:
        raw_parameters = inspect.signature(self.compute).parameters
        parameters = OrderedDict()
        for key in raw_parameters:
            parameters[key] = raw_parameters[key].annotation
        return parameters

    def get_return_type(self) -> Type['Variable']:
        return_type = inspect.signature(self.compute).return_annotation
        return return_type

# The import leaves here for resolving circular import. Also see (1) above line 7
from nodeflow.node.variable import Variable

__all__ = [
    'Function'
]