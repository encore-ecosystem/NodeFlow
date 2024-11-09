from nodeflow import Variable
from nodeflow.adapter import Adapter
from nodeflow.builtin.variables import *


# py2nodeflow adapters
class PyInt2Integer(Adapter):
    def compute(self, variable: int) -> Integer:
        return Integer(variable)

    def is_loses_information(self) -> bool:
        return False


class PyFloat2Float(Adapter):
    def compute(self, variable: float) -> Float:
        return Float(variable)

    def is_loses_information(self) -> bool:
        return False


class PyBool2Boolean(Adapter):
    def compute(self, variable: bool) -> Boolean:
        return Boolean(variable)

    def is_loses_information(self) -> bool:
        return False


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
    'PyBool2Boolean',
    'PyInt2Integer',
    'PyFloat2Float',

    'Boolean2Integer',
    'Integer2Boolean',

    'Integer2Float',
    'Float2Integer',
]