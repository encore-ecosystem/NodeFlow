from typing import Type
from nodeflow.adapters.abstract import Adapter
from nodeflow.node import Int, Bool


class Bool2Int(Adapter):
    @staticmethod
    def convert(bool_node: Bool) -> Int:
        return Int(value=int(bool_node.value))

    @staticmethod
    def get_type_of_source_variable() -> Type[Bool]:
        return Bool

    @staticmethod
    def get_type_of_target_variable() -> Type[Int]:
        return Int

    @staticmethod
    def is_loose_information() -> bool:
        return False


__all__ = [
    "Bool2Int",
]