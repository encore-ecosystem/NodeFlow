from nodeflow.adapter   import *
from nodeflow.converter import *
from nodeflow.node      import *
from nodeflow.dispenser import *

#
# Root Converter
#
import nodeflow.utils as utils

class RootConverter(Converter, utils.Singleton):
    pass

ROOT_CONVERTER = RootConverter()
