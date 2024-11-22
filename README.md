# Nodeflow
<div align="center">
    <img src="img/logo.png"  alt="Warp" />
</div>

Nodeflow if a framework to make pipelines from nodes.

# Status
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/encore-ecosystem/NodeFlow)
![GitHub contributors](https://img.shields.io/github/contributors/encore-ecosystem/NodeFlow)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/encore-ecosystem/NodeFlow/testing.yml)
![PyPI - Downloads](https://img.shields.io/pypi/dm/NodeFlow)
![GitHub License](https://img.shields.io/github/license/encore-ecosystem/NodeFlow)
![GitHub Discussions](https://img.shields.io/github/discussions/encore-ecosystem/NodeFlow)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/NodeFlow)
![GitHub repo size](https://img.shields.io/github/repo-size/encore-ecosystem/NodeFlow)
![GitHub forks](https://img.shields.io/github/forks/encore-ecosystem/NodeFlow)
![GitHub Repo stars](https://img.shields.io/github/stars/encore-ecosystem/NodeFlow)
![GitHub watchers](https://img.shields.io/github/watchers/encore-ecosystem/NodeFlow)
![GitHub Tag](https://img.shields.io/github/v/tag/encore-ecosystem/NodeFlow)

# Usage
<div align="center">
    <img src="https://github.com/encore-ecosystem/NodeFlow/blob/main/img/code_example.svg?raw=trueimg/" alt="Warp" />
</div>

# Installing
Requires python version **3.10 or later**
```bash
pip3 install nodeflow
```
Import it to your code using:
```python
import nodeflow
```

# Types
In `Nodeflow`, everything is a subclass of the `Node` class. 

**Dev note: This is an old documentation. Newest versions of Nodeflow contain adapters from any2any types.**

## Variable
Each node encapsulates information for a specific type and functions as a container.
The code snippet below is taken from the built-in types:
```python
from nodeflow import Variable

class Integer(Variable):
    def __init__(self, value: int):
        super().__init__(value)
    
    # optional
    def __add__(self, other: 'Integer') -> 'Integer':
        return Integer(self.value + other.value)
    
    # optional
    def __mul__(self, other: 'Integer') -> 'Integer':
        return Integer(self.value * other.value)
```

## Function
A `Function` node accepts several `Variable` nodes to perform computations and must return exactly one `Variable`.
You can define your own function in the following manner:
```python
from nodeflow import Function

class Sum(Function):
    def compute(self, a: Integer, b: Integer) -> Integer:
        return a + b
```
However, in practical applications, we recommend using a converter to transform a Python function into a `NodeFlow` function:
```python
from nodeflow import func2node

def sum_of_integers(a: Integer, b: Integer) -> Integer:
    return a + b

Sum = func2node(sum_of_integers)
```
It is **essential** to include all type hints since the `NodeFlow` framework has the capability to derive implicit adapters. 

## Adapter
This is a subclass of `Function`. The computations primarily involve converting the types of `Variable` nodes.
Refer to the examples for clarification. 

## Dispenser
How can we define our pipeline? If the function takes a single parameter, you can follow this pattern: 
```python
my_variable >> my_function
```
If the function requires multiple values, utilize an instance of the `Dispenser`:
```python
from nodeflow import Dispenser, func2node

def increment(a: Integer) -> Integer:
    return Integer(a.value + 1)

def sum_of_integers(a: Integer, b: Integer) -> Integer:
    return a + b

result = Dispenser(
    a = Integer(5) >> func2node(increment),
    b = Integer(10),
) >> func2node(sum_of_integers)
# results.value == 16
```

## Converter
A few words about **deriving implicit adapters**. The core of `NodeFlow` provides a solution for identifying the shortest
path of explicit adapters that can convert one type to another when possible.

The conversion pipeline may sometimes lead to information loss, but the following priority list can guide the process:
- Shortest pipeline without losing information
- Shortest pipeline with information loss
- Long pipeline without losing information **better** than short pipeline with information loss

## Contributing
See `CONTRIBUTING.md` file. 

## Licence
<a href="LICENSE">MIT</a>

## Special thanks
Thank **you** for your interest in the project description!
