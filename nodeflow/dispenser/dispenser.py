from nodeflow.node import Variable, Function


class Dispenser:
    def __init__(self, **kwargs: Variable):
        self.variables_table = kwargs

    def __rshift__(self, other: Function):
        function_types = other.get_parameters()

        # Check for ability to match
        assert len(self.variables_table) == len(function_types)     , "Provided not enough parameters"
        assert self.variables_table.keys() == function_types.keys() , "Provided parameters names doesn't match"

        # Check types
        for key in self.variables_table:
            assert issubclass(type(self.variables_table[key]), function_types[key]), \
                f"Couldn't match key {key}: Expected subclass of {function_types[key]}, but got {type(self.variables_table[key])}"

        return other.compute(**self.variables_table)

__all__ = [
    'Dispenser',
]
