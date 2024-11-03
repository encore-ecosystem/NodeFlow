from nodeflow.builtin.adapters.numeric import Boolean2Integer, Integer2Boolean, Integer2Float, Float2Integer
from nodeflow.converter import Converter


BUILTIN_CONVERTER = Converter(
    adapters = [
        Boolean2Integer(), Integer2Boolean(),
        Integer2Float(), Float2Integer(),
    ]
)


__all__ = [
    'BUILTIN_CONVERTER',
]
