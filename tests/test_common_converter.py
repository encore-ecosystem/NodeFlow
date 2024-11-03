from nodeflow.builtin.adapters.numeric import Integer2Float, Float2Integer, Boolean2Integer
from nodeflow.converter import Converter
from nodeflow.builtin import *


def test_explicit_adapter():
    source_node = Integer(value=15)

    converter = Converter(
        adapters=[Integer2Float(), Float2Integer()],
    )

    assert converter.convert(
        variable=source_node,
        to_type=Float
    ).value == 15.

def test_transitivity():
    source_node = Boolean(value=True)

    converter = Converter(
        adapters=[Boolean2Integer(), Integer2Float()],
    )

    assert converter.convert(
        variable=source_node,
        to_type=Float
    ).value == 1.

