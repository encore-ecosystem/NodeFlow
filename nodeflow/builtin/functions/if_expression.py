from nodeflow import func2node


def py_if(expression, true_case, false_case):
    return true_case if expression else false_case


IF = func2node(py_if)

__all__ = [
    "IF"
]
