from typing import Type
from nodeflow.adapters.abstract import Adapter
from nodeflow.node import Int, Float


class Int2Float(Adapter):
    @staticmethod
    def convert(int_node: Int) -> Float:
        return Float(value=float(int_node.value))

    @staticmethod
    def get_type_of_source_variable() -> Type[Int]:
        return Int

    @staticmethod
    def get_type_of_target_variable() -> Type[Float]:
        return Float

    @staticmethod
    def is_loose_information() -> bool:
        return False


__all__ = [
    "Int2Float",
]