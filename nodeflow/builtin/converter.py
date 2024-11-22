from nodeflow.builtin.adapters import *
from nodeflow.converter import Converter


BUILTIN_CONVERTER = Converter(
    adapters = [
        Boolean2Integer(),
        Integer2Boolean(),

        Integer2Float(),
        Float2Integer(),


        int2Integer(),
        Integer2int(),

        Boolean2bool(),
        bool2Boolean(),

        Float2float(),
        float2Float(),
    ]
)


__all__ = [
    'BUILTIN_CONVERTER',
]
