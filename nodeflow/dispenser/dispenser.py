from nodeflow import Converter, func2node
from nodeflow.node import Variable, Function
from typing import Callable, Union


class Dispenser:
    def __init__(self, **kwargs: object):
        self.variables_table = kwargs

    def __rshift__(self, other: Union[Function|Callable]):
        other = func2node(other) if isinstance(other, Callable) else other
        function_types = other.get_parameters()

        # Check for ability to match
        assert len(self.variables_table) == len(function_types)     , "Provided not enough parameters"
        assert self.variables_table.keys() == function_types.keys() , "Provided parameters names doesn't match"

        # Check types
        for key in self.variables_table:
            # Subclass allowed
            if issubclass(type(self.variables_table[key]), function_types[key]):
                continue

            # Try to find safe pipeline
            assert Converter.ROOT_CONVERTER is not None, "Missing root converter, use context manager to determine it"
            pipeline, is_safe = Converter.ROOT_CONVERTER.get_converting_pipeline(
                source=type(self.variables_table[key]),
                target=function_types[key],
            )

            assert is_safe, f"Couldn't match key {key}: Is there safe adapter {type(self.variables_table[key])} -> {function_types[key]}?"

        return other.compute(**self.variables_table)

__all__ = [
    'Dispenser',
]
