# Nodeflow
----------
Nodeflow if a framework to make pipelines from nodes. You can implement any logic you want by using Variable, 
Function and Adapter nodes. Nodeflow support implicit converter for Variables with priority on non-lossy adapters
converting pipeline 

Installing
----------
Install NodeFlow from PyPI via ``pip install nodeflow``

Usage
----------
To write your own things just implement all required methods in corresponding abstract classes.
Example for Variable:
```python
from nodeflow.node.abstract import Variable


class Bool(Variable):
    def __init__(self, value: bool):
        assert isinstance(value, bool)
        super().__init__(value)
```
