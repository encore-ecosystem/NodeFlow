from nodeflow.adapter import Adapter
from nodeflow.builtin.variables import *


# Boolean <-> Integer
class Boolean2Integer(Adapter):
    def compute(self, variable: Boolean) -> Integer:
        return Integer(value=int(variable.value))

    def is_loses_information(self) -> bool:
        return False

class Integer2Boolean(Adapter):
    def compute(self, variable: Integer) -> Boolean:
        return Boolean(value=bool(variable.value))

    def is_loses_information(self) -> bool:
        return True

# Integer <-> Float
class Integer2Float(Adapter):
    def compute(self, variable: Integer) -> Float:
        return Float(value=float(variable.value))

    def is_loses_information(self) -> bool:
        return False

class Float2Integer(Adapter):
    def compute(self, variable: Float) -> Integer:
        return Integer(value=int(variable.value))

    def is_loses_information(self) -> bool:
        return True


__all__ = [
    'Boolean2Integer',
    'Integer2Boolean',

    'Integer2Float',
    'Float2Integer',
]