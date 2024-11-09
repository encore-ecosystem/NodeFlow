from nodeflow.builtin.adapters  import PyInt2Integer, PyBool2Boolean, PyFloat2Float
from nodeflow.builtin.variables import Integer, Boolean, Float


def test_pyint2int():
    assert PyInt2Integer().compute(5) == Integer(5)

def test_pybool2bool():
    assert PyBool2Boolean().compute(True) == Boolean(True)
    assert PyBool2Boolean().compute(False) == Boolean(False)

def test_pyfloat2float():
    assert PyFloat2Float().compute(5.) == Float(5.)
