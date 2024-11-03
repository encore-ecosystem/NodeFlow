# Register ROOT Converter
from nodeflow import ROOT_CONVERTER
from nodeflow.builtin import Integer, Boolean, Float
from nodeflow.builtin.converter import BUILTIN_CONVERTER
from nodeflow import Dispenser, func2node
from typing import Union

ROOT_CONVERTER.register_converter(BUILTIN_CONVERTER)
# End

Numeric = Union[Boolean, Integer, Float]
def generic_add(lhs: Numeric, rhs: Numeric) -> Numeric:
    return lhs + rhs


def test_generic_auto_converting():
    a = Integer(value=5)
    b = Float(value=-5.)

    pipeline = Dispenser(
        lhs=a,
        rhs=b,
    ) >> func2node(generic_add)
    assert pipeline.value == 0
