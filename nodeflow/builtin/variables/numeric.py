from nodeflow.node.variable import Variable


class Boolean(Variable):
    def __init__(self, value: bool):
        super().__init__(value)

    def __add__(self, other: 'Boolean') -> 'Boolean':
        return Boolean(self.value + other.value)

    def __mul__(self, other: 'Boolean') -> 'Boolean':
        return Boolean(self.value * other.value)

class Integer(Variable):
    def __init__(self, value: int):
        super().__init__(value)

    def __add__(self, other: 'Integer') -> 'Integer':
        return Integer(self.value + other.value)

    def __mul__(self, other: 'Integer') -> 'Integer':
        return Integer(self.value * other.value)

class Float(Variable):
    def __init__(self, value: float):
        super().__init__(value)

    def __add__(self, other: 'Float') -> 'Float':
        return Float(self.value + other.value)

    def __mul__(self, other: 'Float') -> 'Float':
        return Float(self.value * other.value)

__all__ = [
    'Boolean',
    'Integer',
    'Float',
]
