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

Design goals:
- Clear shape semantics and explicit dimension handling.
- Consistent arithmetic and indexing behavior across containers.
- Interoperable C++ and Python usage with low-overhead bindings.

Typical usage:
        >>> from stratax import Tensor, zeros, to_matrix
        >>> t = zeros([2, 3])
        >>> m = to_matrix(t)
"""

from ._core import __version__, __doc__ as __core_doc__, __author__
from .shape import Shape
from .tensor import Tensor
from .vector import Vector
from .matrix import Matrix
from .conversions import to_matrix, to_tensor, to_vector
from .creation import full, identity, ones, zeros
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
    "StrataxError",
    "ShapeError",
    "DimensionError",
    "IndexError",
    "TypeError",
    "BroadcastError",
    "ZeroDivisionError",
]
