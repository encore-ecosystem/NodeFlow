from nodeflow.converter import func2node
from nodeflow.dispenser import Dispenser
from nodeflow.builtin import Float

from vector_variable import *


def scalar_product(first_vector: Vector3, second_vector: Vector3) -> Float:
    return Float(value = sum(first_vector*second_vector))

def normalize(a: Vector3) -> Vector3:
    import math
    norm = math.sqrt(sum(a*a))
    return Vector3(
        value = [x/norm for x in a]
    )

def main():
    a = Vector3([1, 2, 3])
    b = Vector3([3, 2, 1])

    result = Dispenser(
        first_vector  = a >> func2node(normalize),
        second_vector = b >> func2node(normalize),
    ) >> func2node(scalar_product)


    print(result.value)


if __name__ == "__main__":
    main()
