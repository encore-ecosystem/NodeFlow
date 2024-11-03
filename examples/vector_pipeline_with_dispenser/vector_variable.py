from nodeflow.node import Variable


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


__all__ = [
    'Vector3'
]
