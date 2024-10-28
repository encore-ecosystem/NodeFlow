from nodeflow.converter import Converter
from nodeflow.node import *
from nodeflow.adapters import *
from nodeflow.node.variable import Variable

# =========
# Variables
# =========
class Int(Variable):
    def __init__(self, value: int):
        assert isinstance(value, int)
        super().__init__(value)

class Float(Variable):
    def __init__(self, value: float):
        assert isinstance(value, float)
        super().__init__(value)

class Bool(Variable):
    def __init__(self, value: bool):
        assert isinstance(value, bool)
        super().__init__(value)

# ========
# Adapters
# ========
class Int2Float(Adapter):
    def convert(self, variable: Int) -> Float:
        return Float(value=float(variable.value))

    def is_loose_information(self) -> bool:
        return False

class Float2Int(Adapter):
    def convert(self, variable: Float) -> Int:
        return Int(value=int(variable.value))

    def is_loose_information(self) -> bool:
        return True

class Bool2Int(Adapter):
    def convert(self, variable: Bool) -> Int:
        return Int(value=int(variable.value))

    def is_loose_information(self) -> bool:
        return False


def test_int2float():
    source_node = Int(value=15)

    converter = Converter(
        adapters=[Int2Float(), Float2Int()],
    )

    assert converter.convert(
        variable=source_node,
        to_type=Float
    ).value == 15.

def test_bool2float():
    source_node = Bool(value=True)

    converter = Converter(
        adapters=[Bool2Int(), Int2Float(), Float2Int()],
    )

    assert converter.convert(
        variable=source_node,
        to_type=Float
    ).value == 1.

test_bool2float()
