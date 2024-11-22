from nodeflow import Variable
from nodeflow.builtin.variables import Integer, Float, Boolean
from nodeflow.adapter import Adapter
from abc import ABCMeta


class PythonicAdapter(Adapter, metaclass=ABCMeta):
    def is_loses_information(self) -> bool:
        return False



class int2Integer(PythonicAdapter):
    def compute(self, variable: int) -> Integer:
        return Integer(variable)

class Integer2int(PythonicAdapter):
    def compute(self, variable: Integer) -> int:
        return variable.value


class float2Float(PythonicAdapter):
    def compute(self, variable: float) -> Float:
        return Float(variable)

class Float2float(PythonicAdapter):
    def compute(self, variable: Float) -> float:
        return variable.value


class bool2Boolean(PythonicAdapter):
    def compute(self, variable: bool) -> Boolean:
        return Boolean(variable)

class Boolean2bool(PythonicAdapter):
    def compute(self, variable: Boolean) -> bool:
        return variable.value


__all__ = [
    'int2Integer',
    'Integer2int',

    'float2Float',
    'Float2float',

    'bool2Boolean',
    'Boolean2bool',
]
