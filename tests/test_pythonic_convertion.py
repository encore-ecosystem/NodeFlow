from nodeflow import Dispenser
from nodeflow.builtin import Integer

def add_integers(a: int, b: int) -> Integer:
    return Integer(a + b)

def add_different_types(a: int, b: Integer) -> int:
    return a + b.value

def test_integers():
    assert (Dispenser(a = 5, b = 2) >> add_integers) == 7

def test_different_types():
    assert (Dispenser(a = 5, b = Integer(2)) >> add_different_types) == 7
