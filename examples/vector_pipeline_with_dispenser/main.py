from nodeflow.node import Variable
from nodeflow.converter import func2node
from nodeflow.dispenser import Dispenser


class Vector3(Variable):
    def __init__(self, value: list[float]):
        super().__init__(value)

    def __mul__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(
            value = [a_i*b_i for a_i, b_i in zip(self.value, other.value)]
        )

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, i) -> float:
        return self.value[i]

    def __iter__(self):
        return self.value.__iter__()

class Float(Variable):
    def __init__(self, value: float):
        super().__init__(value)


def scalar_product(a: Vector3, b: Vector3) -> Float:
    return Float(value = sum(a*b))

def normalize(a: Vector3) -> Vector3:
    import math
    norm = math.sqrt(sum(a*a))
    return Vector3(
        value = [x/norm for x in a]
    )

def main():
    a = Vector3([1, 1, 1])
    b = Vector3([2, 2, 2])

    scalar_product_node = func2node(scalar_product)
    normalize_node      = func2node(normalize)

    res = Dispenser(
        normalize_node.compute(a),
        normalize_node.compute(b),
    ) >> scalar_product_node
    print(res.value)


if __name__ == "__main__":
    main()
