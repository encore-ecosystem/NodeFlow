from abc import ABC, abstractmethod
from typing import Type
from nodeflow.node.variable import Variable
import inspect



class Adapter(ABC):
    @abstractmethod
    def convert(self, variable: Variable):
        raise NotImplementedError

    def get_type_of_source_variable(self) -> Type[Variable]:
        signature = inspect.signature(self.convert)
        return signature.parameters['variable'].annotation

    def get_type_of_target_variable(self) -> Type[Variable]:
        signature = inspect.signature(self.convert)
        return signature.return_annotation

    @abstractmethod
    def is_loses_information(self) -> bool:
        raise NotImplementedError


__all__ = [
    'Adapter'
]