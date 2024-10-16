from nodeflow.converter import Converter
from nodeflow.node import *
from nodeflow.adapters import *


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
