from nodeflow.node import Variable, Function


class Dispenser:
    def __init__(self, *args):
        # check types
        for arg in args:
            assert isinstance(arg, Variable)

        self.variables = args

    def __rshift__(self, other: Function):
        return other.compute(*self.variables)

__all__ = [
    'Dispenser',
]
