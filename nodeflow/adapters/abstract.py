from abc import ABC, abstractmethod
from nodeflow.node.abstract import Variable
from typing import Type


class Adapter(ABC):
    @staticmethod
    @abstractmethod
    def convert(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_type_of_source_variable() -> Type[Variable]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_type_of_target_variable() -> Type[Variable]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_loose_information() -> bool:
        raise NotImplementedError


__all__ = [
    'Adapter'
]