"""Stratax Python API.

Stratax is a scientific array library centered on a modern C++20 core with a
clean Python interface. The package provides strongly shaped array containers
and predictable row-major behavior for numeric workflows.

What this package includes:
- Shape metadata with validation via Shape.
- Dense containers: Vector (1D), Matrix (2D), and Tensor (N-D).
- Conversion helpers between container kinds.
- Creation helpers for common initialization patterns.
- Domain-specific exception types for shape, indexing, typing, and arithmetic
  errors.
"""

from ._core import __author__, __version__
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
from .matrix import Matrix
from .shape import Shape
from .tensor import Tensor
from .vector import Vector

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
