from nodeflow.node.variable import Variable
import pathlib


class PathVariable(Variable):
    def __init__(self, value: pathlib.Path):
        assert isinstance(value, pathlib.Path)
        assert value.exists()
        super().__init__(value.resolve())

    def __truediv__(self, other: str) -> pathlib.Path:
        return self.value / other


__all__ = [
    'PathVariable',
]
