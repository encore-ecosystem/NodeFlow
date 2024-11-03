from nodeflow.node.variable import Variable


class Result(Variable):
    def __init__(self, value: bool):
        assert isinstance(value, bool)
        super().__init__(value)


__all__ = [
    'Result'
]
