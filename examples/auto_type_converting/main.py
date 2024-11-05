from nodeflow.builtin import Integer, Float
from nodeflow import Dispenser, func2node
from nodeflow.builtin.converter import BUILTIN_CONVERTER


def add(lhs: Float, rhs: Float) -> Float:
    return lhs + rhs


def main():
    a = Integer(value=5)
    b = Float(value=-5.)

    with BUILTIN_CONVERTER:
        pipeline = Dispenser(
            lhs=a,
            rhs=b,
        ) >> func2node(add)

    assert pipeline.value == 0

if __name__ == "__main__":
    main()
