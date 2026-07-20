"""Typing stubs for the compiled Stratax ``_core`` extension.

This module exposes low-level container types, conversion/creation helpers,
and exception classes implemented in the native extension.
"""

from collections.abc import Iterable, Iterator
from typing import overload


class Shape:
    """C++ shape object storing non-negative array dimensions."""

    @overload
    def __init__(self) -> None:
        """Create an empty rank-0 shape."""
        ...

    @overload
    def __init__(self, other: "Shape") -> None:
        """Copy another shape."""
        ...

    @overload
    def __init__(self, dims: Iterable[int]) -> None:
        """Create a shape from non-negative integer dimensions."""
        ...

    def rank(self) -> int:
        """Return the number of dimensions."""
        ...

    def elements(self) -> int:
        """Return the product of all dimensions, or zero for rank 0."""
        ...

    def empty(self) -> bool:
        """Return whether the shape has rank 0."""
        ...

    def __len__(self) -> int:
        """Return the rank."""
        ...

    def __getitem__(self, index: int) -> int:
        """Return one dimension by zero-based index."""
        ...

    def __iter__(self) -> Iterator[int]:
        """Iterate over dimensions in order."""
        ...

    def __eq__(self, other: object) -> bool:
        """Return whether two shapes have identical dimensions."""
        ...

    def __ne__(self, other: object) -> bool:
        """Return whether two shapes differ."""
        ...

    def __repr__(self) -> str:
        """Return a tuple-style shape representation."""
        ...


class Vector:
    """Rank-1 floating-point array with a non-negative size."""

    @overload
    def __init__(self) -> None:
        """Create an empty vector."""
        ...

    @overload
    def __init__(self, size: int) -> None:
        """Create a vector with size default-initialized elements."""
        ...

    @overload
    def __init__(self, shape: _Shape) -> None:
        """Create a vector from a rank-1 shape."""
        ...

    @overload
    def __init__(self, other: "Vector") -> None:
        """Copy another vector."""
        ...

    @overload
    def __init__(self, size: int, value: float) -> None:
        """Create a vector and fill every element with value."""
        ...

    @overload
    def __init__(self, values: Iterable[float]) -> None:
        """Create a vector from numeric values."""
        ...

    def size(self) -> int:
        """Return the number of elements."""
        ...

    def rank(self) -> int:
        """Return the vector rank."""
        ...

    def empty(self) -> bool:
        """Return whether the vector has no elements."""
        ...

    def shape(self) -> _Shape:
        """Return the vector shape."""
        ...

    def strides(self) -> list[int]:
        """Return row-major stride metadata."""
        ...

    def fill(self, value: float) -> None:
        """Assign value to every element."""
        ...

    def tolist(self) -> list[float]:
        """Return element values as a Python list."""
        ...

    def reshape(self, shape: _Shape | Iterable[int]) -> "Tensor":
        """Return a reshaped tensor."""
        ...

    def flatten(self) -> "Vector":
        """Return a flattened vector."""
        ...

    def __len__(self) -> int:
        """Return the number of elements."""
        ...

    @overload
    def __getitem__(self, index: int) -> float:
        """Return an element by non-negative flat index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> "Vector":
        """Return a sliced vector using Python slice syntax."""
        ...

    def __setitem__(self, index: int, value: float) -> None:
        """Set an element by non-negative flat index."""
        ...

    def __iter__(self) -> Iterator[float]:
        """Iterate over element values."""
        ...

    def __eq__(self, other: object) -> bool:
        """Return whether two vectors have the same shape and values."""
        ...

    def __ne__(self, other: object) -> bool:
        """Return whether two vectors differ."""
        ...

    def __add__(self, other: "Vector | float") -> "Vector":
        """Return elementwise vector or scalar addition."""
        ...

    def __radd__(self, other: float) -> "Vector":
        """Return scalar plus vector."""
        ...

    def __sub__(self, other: "Vector | float") -> "Vector":
        """Return elementwise vector or scalar subtraction."""
        ...

    def __rsub__(self, other: float) -> "Vector":
        """Return scalar minus vector."""
        ...

    def __mul__(self, other: "Vector | float") -> "Vector":
        """Return elementwise vector or scalar multiplication."""
        ...

    def __rmul__(self, other: float) -> "Vector":
        """Return scalar times vector."""
        ...

    def __truediv__(self, other: "Vector | float") -> "Vector":
        """Return elementwise vector or scalar division."""
        ...

    def __rtruediv__(self, other: float) -> "Vector":
        """Return scalar divided by vector elements."""
        ...

    def __iadd__(self, other: "Vector | float") -> "Vector":
        """Add a vector or scalar into this vector."""
        ...

    def __isub__(self, other: "Vector | float") -> "Vector":
        """Subtract a vector or scalar from this vector."""
        ...

    def __imul__(self, other: "Vector | float") -> "Vector":
        """Multiply this vector by a vector or scalar."""
        ...

    def __itruediv__(self, other: "Vector | float") -> "Vector":
        """Divide this vector by a vector or scalar."""
        ...

    def __pos__(self) -> "Vector":
        """Return a copy of this vector."""
        ...

    def __neg__(self) -> "Vector":
        """Return a vector with every element negated."""
        ...

    def __repr__(self) -> str:
        """Return a compact vector representation."""
        ...


class Matrix:
    """Rank-2 row-major floating-point array with non-negative dimensions."""

    @overload
    def __init__(self) -> None:
        """Create an empty 0 by 0 matrix."""
        ...

    @overload
    def __init__(self, rows: int, cols: int) -> None:
        """Create a matrix with default-initialized elements."""
        ...

    @overload
    def __init__(self, shape: _Shape) -> None:
        """Create a matrix from a rank-2 shape."""
        ...

    @overload
    def __init__(self, other: "Matrix") -> None:
        """Copy another matrix."""
        ...

    @overload
    def __init__(self, rows: int, cols: int, value: float) -> None:
        """Create a matrix and fill every element with value."""
        ...

    @overload
    def __init__(self, values: Iterable[Iterable[float]]) -> None:
        """Create a matrix from equal-length rows of numeric values."""
        ...

    def size(self) -> int:
        """Return the total number of elements."""
        ...

    def rank(self) -> int:
        """Return the matrix rank."""
        ...

    def empty(self) -> bool:
        """Return whether the matrix has no elements."""
        ...

    def rows(self) -> int:
        """Return the number of rows."""
        ...

    def cols(self) -> int:
        """Return the number of columns."""
        ...

    def shape(self) -> _Shape:
        """Return the matrix shape."""
        ...

    def strides(self) -> list[int]:
        """Return row-major stride metadata."""
        ...

    def fill(self, value: float) -> None:
        """Assign value to every element."""
        ...

    def tolist(self) -> list[list[float]]:
        """Return values as nested row lists."""
        ...

    def reshape(self, shape: _Shape | Iterable[int]) -> "Tensor":
        """Return a reshaped tensor."""
        ...

    def flatten(self) -> Vector:
        """Return matrix data as a flattened vector."""
        ...

    def __len__(self) -> int:
        """Return the number of rows."""
        ...

    @overload
    def __getitem__(self, index: tuple[int, int]) -> float:
        """Return an element by non-negative (row, column)."""
        ...

    @overload
    def __getitem__(self, index: slice | tuple[int | slice, int | slice]) -> "Matrix":
        """Return a matrix slice using Python slice syntax."""
        ...

    def __setitem__(self, index: tuple[int, int], value: float) -> None:
        """Set an element by non-negative (row, column)."""
        ...

    def __iter__(self) -> Iterator[float]:
        """Iterate over elements in row-major order."""
        ...

    def __eq__(self, other: object) -> bool:
        """Return whether two matrices have the same shape and values."""
        ...

    def __ne__(self, other: object) -> bool:
        """Return whether two matrices differ."""
        ...

    def __add__(self, other: "Matrix | float") -> "Matrix":
        """Return elementwise matrix or scalar addition."""
        ...

    def __radd__(self, other: float) -> "Matrix":
        """Return scalar plus matrix."""
        ...

    def __sub__(self, other: "Matrix | float") -> "Matrix":
        """Return elementwise matrix or scalar subtraction."""
        ...

    def __rsub__(self, other: float) -> "Matrix":
        """Return scalar minus matrix."""
        ...

    def __mul__(self, other: "Matrix | float") -> "Matrix":
        """Return elementwise matrix or scalar multiplication."""
        ...

    def __rmul__(self, other: float) -> "Matrix":
        """Return scalar times matrix."""
        ...

    def __truediv__(self, other: "Matrix | float") -> "Matrix":
        """Return elementwise matrix or scalar division."""
        ...

    def __rtruediv__(self, other: float) -> "Matrix":
        """Return scalar divided by matrix elements."""
        ...

    def __iadd__(self, other: "Matrix | float") -> "Matrix":
        """Add a matrix or scalar into this matrix."""
        ...

    def __isub__(self, other: "Matrix | float") -> "Matrix":
        """Subtract a matrix or scalar from this matrix."""
        ...

    def __imul__(self, other: "Matrix | float") -> "Matrix":
        """Multiply this matrix by a matrix or scalar."""
        ...

    def __itruediv__(self, other: "Matrix | float") -> "Matrix":
        """Divide this matrix by a matrix or scalar."""
        ...

    def __pos__(self) -> "Matrix":
        """Return a copy of this matrix."""
        ...

    def __neg__(self) -> "Matrix":
        """Return a matrix with every element negated."""
        ...

    def __repr__(self) -> str:
        """Return a compact matrix representation."""
        ...


class Tensor:
    """N-dimensional row-major floating-point array with non-negative dimensions."""

    @overload
    def __init__(self) -> None:
        """Create an empty rank-0 tensor."""
        ...

    @overload
    def __init__(self, shape: _Shape) -> None:
        """Create a tensor from a shape."""
        ...

    @overload
    def __init__(self, shape: _Shape, value: float) -> None:
        """Create a tensor and fill every element with value."""
        ...

    @overload
    def __init__(self, other: "Tensor") -> None:
        """Copy another tensor."""
        ...

    @overload
    def __init__(self, shape: Iterable[int]) -> None:
        """Create a tensor from non-negative integer dimensions."""
        ...

    @overload
    def __init__(self, shape: Iterable[int], value: float) -> None:
        """Create a tensor from dimensions and fill it with value."""
        ...

    def size(self) -> int:
        """Return the total number of elements."""
        ...

    def rank(self) -> int:
        """Return the tensor rank."""
        ...

    def empty(self) -> bool:
        """Return whether the tensor has no elements."""
        ...

    def shape(self) -> _Shape:
        """Return the tensor shape."""
        ...

    def strides(self) -> list[int]:
        """Return row-major stride metadata."""
        ...

    def fill(self, value: float) -> None:
        """Assign value to every element."""
        ...

    def tolist(self) -> list[float]:
        """Return values as a flat Python list."""
        ...

    def reshape(self, shape: _Shape | Iterable[int]) -> "Tensor":
        """Return a tensor view with the requested shape."""
        ...

    def flatten(self) -> Vector:
        """Return tensor data as a rank-1 vector."""
        ...

    def __len__(self) -> int:
        """Return the number of flat elements."""
        ...

    @overload
    def __getitem__(self, index: int | tuple[int, ...]) -> float:
        """Return an element by non-negative flat index or multidimensional integer tuple."""
        ...

    @overload
    def __getitem__(self, index: slice | tuple[int | slice, ...]) -> "Tensor":
        """Return a tensor slice using Python slice syntax."""
        ...

    def __setitem__(self, index: int | tuple[int, ...], value: float) -> None:
        """Set an element by non-negative flat index or multidimensional tuple."""
        ...

    def __iter__(self) -> Iterator[float]:
        """Iterate over elements in row-major order."""
        ...

    def __eq__(self, other: object) -> bool:
        """Return whether two tensors have the same shape and values."""
        ...

    def __ne__(self, other: object) -> bool:
        """Return whether two tensors differ."""
        ...

    def __add__(self, other: "Tensor | float") -> "Tensor":
        """Return elementwise tensor or scalar addition."""
        ...

    def __radd__(self, other: float) -> "Tensor":
        """Return scalar plus tensor."""
        ...

    def __sub__(self, other: "Tensor | float") -> "Tensor":
        """Return elementwise tensor or scalar subtraction."""
        ...

    def __rsub__(self, other: float) -> "Tensor":
        """Return scalar minus tensor."""
        ...

    def __mul__(self, other: "Tensor | float") -> "Tensor":
        """Return elementwise tensor or scalar multiplication."""
        ...

    def __rmul__(self, other: float) -> "Tensor":
        """Return scalar times tensor."""
        ...

    def __truediv__(self, other: "Tensor | float") -> "Tensor":
        """Return elementwise tensor or scalar division."""
        ...

    def __rtruediv__(self, other: float) -> "Tensor":
        """Return scalar divided by tensor elements."""
        ...

    def __iadd__(self, other: "Tensor | float") -> "Tensor":
        """Add a tensor or scalar into this tensor."""
        ...

    def __isub__(self, other: "Tensor | float") -> "Tensor":
        """Subtract a tensor or scalar from this tensor."""
        ...

    def __imul__(self, other: "Tensor | float") -> "Tensor":
        """Multiply this tensor by a tensor or scalar."""
        ...

    def __itruediv__(self, other: "Tensor | float") -> "Tensor":
        """Divide this tensor by a tensor or scalar."""
        ...

    def __pos__(self) -> "Tensor":
        """Return a copy of this tensor."""
        ...

    def __neg__(self) -> "Tensor":
        """Return a tensor with every element negated."""
        ...

    def __repr__(self) -> str:
        """Return a compact tensor representation."""
        ...


def to_vector(arr: Vector | Matrix | Tensor) -> Vector:
    """Convert an array-like object to a vector."""
    ...


def to_matrix(arr: Vector | Matrix | Tensor) -> Matrix:
    """Convert an array-like object to a matrix."""
    ...


def to_tensor(arr: Vector | Matrix | Tensor) -> Tensor:
    """Convert an array-like object to a tensor."""
    ...


def zeros(shape: _Shape) -> Tensor:
    """Create a tensor filled with zeros."""
    ...


def ones(shape: _Shape) -> Tensor:
    """Create a tensor filled with ones."""
    ...


def full(shape: _Shape, value: float) -> Tensor:
    """Create a tensor filled with a constant value."""
    ...


def identity(size: int) -> Tensor:
    """Create a square identity tensor."""
    ...


class StrataxError(RuntimeError):
    """Base class for Stratax runtime errors."""


class ShapeError(StrataxError):
    """Raised when a shape is invalid or incompatible."""


class DimensionError(StrataxError):
    """Raised when a dimension value or rank is invalid."""


class IndexError(StrataxError):
    """Raised when an array index is out of bounds."""


class TypeError(StrataxError):
    """Raised when a Stratax API receives an unsupported type."""


class BroadcastError(StrataxError):
    """Raised when shapes cannot be broadcast together."""


class ZeroDivisionError(StrataxError):
    """Raised when a Stratax operation divides by zero."""
