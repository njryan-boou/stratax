from ._core import __version__, __doc__ as __core_doc__, __author__
from .shape import Shape
from .tensor import Tensor
from .vector import Vector
from .matrix import Matrix
from .conversions import to_matrix, to_tensor, to_vector
from .creation import full, identity, ones, zeros
from .reductions import argmax, argmin, max, mean, min, prod, std, sum, var
from .exceptions import (
    BroadcastError,
    DimensionError,
    IndexError,
    ShapeError,
    StrataxError,
    TypeError,
    ZeroDivisionError,
)

__all__ = [
    "Shape",
    "Tensor",
    "Vector",
    "Matrix",
    "to_vector",
    "to_matrix",
    "to_tensor",
    "zeros",
    "ones",
    "full",
    "identity",
    "sum",
    "prod",
    "max",
    "min",
    "argmax",
    "argmin",
    "mean",
    "var",
    "std",
    "StrataxError",
    "ShapeError",
    "DimensionError",
    "IndexError",
    "TypeError",
    "BroadcastError",
    "ZeroDivisionError",
]
