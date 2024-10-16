from typing import Type
from nodeflow.adapters.abstract import Adapter
from nodeflow.node import Int, Float


class Float2Int(Adapter):
    @staticmethod
    def convert(float_node: Float) -> Int:
        return Int(value=int(float_node.value))

    @staticmethod
    def get_type_of_source_variable() -> Type[Float]:
        return Float

    @staticmethod
    def get_type_of_target_variable() -> Type[Int]:
        return Int

    @staticmethod
    def is_loose_information() -> bool:
        return True


__all__ = [
    "Float2Int",
]