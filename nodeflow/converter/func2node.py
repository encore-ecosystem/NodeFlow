from typing import Callable
from nodeflow.node import Function, Node
import inspect


class ConvertedFunction(Function):
    def compute(self, *args, **kwargs) -> Node:
        raise NotImplementedError


def func2node(func: Callable) -> Function:
    function_signature = inspect.signature(func)

    function = ConvertedFunction()
    function.compute = func
    return function


__all__ = [
    "func2node"
]