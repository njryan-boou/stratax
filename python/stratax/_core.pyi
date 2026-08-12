"""Stratax Python API."""

from collections.abc import Iterable, Iterator
from typing import overload

__author__: str
__doc__: str | None
__version__: str


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
    def __init__(self, size: int) -> None:
        """Create a single-dimension shape."""
        ...

    @overload
    def __init__(self, dims: Iterable[int]) -> None:
        """Create a shape from non-negative integer dimensions."""
        ...

    @property
    def rank(self) -> int:
        """Return the number of dimensions."""
        ...

    @property
    def elements(self) -> int:
        """Return the product of all dimensions, or zero for rank 0."""
        ...

    @property
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
    def __init__(self, value: int | Iterable[float] | "Vector") -> None:
        """Create a vector from a size, iterable, or vector."""
        ...

    @overload
    def __init__(self, size: int, value: float) -> None:
        """Create a vector and fill every element with value."""
        ...

    @property
    def size(self) -> int:
        """Return the number of elements."""
        ...

    @property
    def rank(self) -> int:
        """Return the vector rank."""
        ...

    @property
    def empty(self) -> bool:
        """Return whether the vector has no elements."""
        ...

    @property
    def shape(self) -> Shape:
        """Return the vector shape."""
        ...

    @property
    def strides(self) -> list[int]:
        """Return row-major stride metadata."""
        ...

    def fill(self, value: float) -> None:
        """Assign value to every element."""
        ...

    def tolist(self) -> list[float]:
        """Return element values as a Python list."""
        ...

    def reshape(self, shape: Shape | Iterable[int]) -> "Tensor":
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
        """Return an element by flat index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> "Vector":
        """Return a sliced vector."""
        ...

    def __setitem__(self, index: int, value: float) -> None:
        """Set an element by flat index."""
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
        """Return element-wise vector or scalar addition."""
        ...

    def __radd__(self, other: float) -> "Vector":
        """Return scalar plus vector."""
        ...

    def __sub__(self, other: "Vector | float") -> "Vector":
        """Return element-wise vector or scalar subtraction."""
        ...

    def __rsub__(self, other: float) -> "Vector":
        """Return scalar minus vector."""
        ...

    def __mul__(self, other: "Vector | float") -> "Vector":
        """Return element-wise vector or scalar multiplication."""
        ...

    def __rmul__(self, other: float) -> "Vector":
        """Return scalar times vector."""
        ...

    def __truediv__(self, other: "Vector | float") -> "Vector":
        """Return element-wise vector or scalar division."""
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
    def __init__(self, value: Iterable[Iterable[float]] | "Matrix") -> None:
        """Create a matrix from rows or a matrix."""
        ...

    @overload
    def __init__(self, rows: int, cols: int, value: float) -> None:
        """Create a matrix and fill every element with value."""
        ...

    @property
    def size(self) -> int:
        """Return the total number of elements."""
        ...

    @property
    def rank(self) -> int:
        """Return the matrix rank."""
        ...

    @property
    def empty(self) -> bool:
        """Return whether the matrix has no elements."""
        ...

    @property
    def rows(self) -> int:
        """Return the number of rows."""
        ...

    @property
    def cols(self) -> int:
        """Return the number of columns."""
        ...

    @property
    def shape(self) -> Shape:
        """Return the matrix shape."""
        ...

    @property
    def strides(self) -> list[int]:
        """Return row-major stride metadata."""
        ...

    def fill(self, value: float) -> None:
        """Assign value to every element."""
        ...

    def tolist(self) -> list[list[float]]:
        """Return values as nested row lists."""
        ...

    def reshape(self, shape: Shape | Iterable[int]) -> "Tensor":
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
        """Return an element by row and column."""
        ...

    @overload
    def __getitem__(self, index: slice) -> "Matrix":
        """Return a row slice."""
        ...

    @overload
    def __getitem__(self, index: tuple[int, slice]) -> "Matrix":
        """Return a matrix slice."""
        ...

    @overload
    def __getitem__(self, index: tuple[slice, int | slice]) -> "Matrix":
        """Return a matrix slice."""
        ...

    def __setitem__(self, index: tuple[int, int], value: float) -> None:
        """Set an element by row and column."""
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
        """Return element-wise matrix or scalar addition."""
        ...

    def __radd__(self, other: float) -> "Matrix":
        """Return scalar plus matrix."""
        ...

    def __sub__(self, other: "Matrix | float") -> "Matrix":
        """Return element-wise matrix or scalar subtraction."""
        ...

    def __rsub__(self, other: float) -> "Matrix":
        """Return scalar minus matrix."""
        ...

    def __mul__(self, other: "Matrix | float") -> "Matrix":
        """Return element-wise matrix or scalar multiplication."""
        ...

    def __rmul__(self, other: float) -> "Matrix":
        """Return scalar times matrix."""
        ...

    def __truediv__(self, other: "Matrix | float") -> "Matrix":
        """Return element-wise matrix or scalar division."""
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
    def __init__(self, shape: Shape | Iterable[int] | "Tensor") -> None:
        """Create a tensor from a shape, dimensions, or tensor."""
        ...

    @overload
    def __init__(self, shape: Shape | Iterable[int], value: float) -> None:
        """Create a tensor and fill every element with value."""
        ...

    @property
    def size(self) -> int:
        """Return the total number of elements."""
        ...

    @property
    def rank(self) -> int:
        """Return the tensor rank."""
        ...

    @property
    def empty(self) -> bool:
        """Return whether the tensor has no elements."""
        ...

    @property
    def shape(self) -> Shape:
        """Return the tensor shape."""
        ...

    @property
    def strides(self) -> list[int]:
        """Return row-major stride metadata."""
        ...

    def fill(self, value: float) -> None:
        """Assign value to every element."""
        ...

    def tolist(self) -> list[float]:
        """Return values as a flat Python list."""
        ...

    def reshape(self, shape: Shape | Iterable[int]) -> "Tensor":
        """Return a tensor with the requested shape."""
        ...

    def flatten(self) -> Vector:
        """Return tensor data as a rank-1 vector."""
        ...

    def __len__(self) -> int:
        """Return the number of flat elements."""
        ...

    def __getitem__(self, index: int | slice | tuple[int | slice, ...]) -> float | "Tensor":
        """Return an element or tensor slice."""
        ...

    def __setitem__(self, index: int | tuple[int, ...], value: float) -> None:
        """Set an element by flat or multidimensional index."""
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
        """Return element-wise tensor or scalar addition."""
        ...

    def __radd__(self, other: float) -> "Tensor":
        """Return scalar plus tensor."""
        ...

    def __sub__(self, other: "Tensor | float") -> "Tensor":
        """Return element-wise tensor or scalar subtraction."""
        ...

    def __rsub__(self, other: float) -> "Tensor":
        """Return scalar minus tensor."""
        ...

    def __mul__(self, other: "Tensor | float") -> "Tensor":
        """Return element-wise tensor or scalar multiplication."""
        ...

    def __rmul__(self, other: float) -> "Tensor":
        """Return scalar times tensor."""
        ...

    def __truediv__(self, other: "Tensor | float") -> "Tensor":
        """Return element-wise tensor or scalar division."""
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


def zeros(shape: Shape) -> Tensor:
    """Create a tensor filled with zeros."""
    ...


def ones(shape: Shape) -> Tensor:
    """Create a tensor filled with ones."""
    ...


def full(shape: Shape, value: float) -> Tensor:
    """Create a tensor filled with a constant value."""
    ...


def identity(size: int) -> Tensor:
    """Create a square identity tensor."""
    ...


@overload
def sum(arr: Vector | Matrix | Tensor) -> float:
    """Return the sum of all elements."""
    ...


@overload
def sum(arr: Vector | Matrix | Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Return sums along an axis."""
    ...


@overload
def prod(arr: Vector | Matrix | Tensor) -> float:
    """Return the product of all elements."""
    ...


@overload
def prod(arr: Vector | Matrix | Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Return products along an axis."""
    ...


@overload
def max(arr: Vector | Matrix | Tensor) -> float:
    """Return the largest element."""
    ...


@overload
def max(arr: Vector | Matrix | Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Return maxima along an axis."""
    ...


@overload
def min(arr: Vector | Matrix | Tensor) -> float:
    """Return the smallest element."""
    ...


@overload
def min(arr: Vector | Matrix | Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Return minima along an axis."""
    ...


@overload
def argmax(arr: Vector | Matrix | Tensor) -> int:
    """Return the flat index of the largest element."""
    ...


@overload
def argmax(arr: Vector | Matrix | Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Return argmax indices along an axis."""
    ...


@overload
def argmin(arr: Vector | Matrix | Tensor) -> int:
    """Return the flat index of the smallest element."""
    ...


@overload
def argmin(arr: Vector | Matrix | Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Return argmin indices along an axis."""
    ...


@overload
def mean(arr: Vector | Matrix | Tensor) -> float:
    """Return the arithmetic mean."""
    ...


@overload
def mean(arr: Vector | Matrix | Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Return means along an axis."""
    ...


@overload
def var(arr: Vector | Matrix | Tensor) -> float:
    """Return the population variance."""
    ...


@overload
def var(arr: Vector | Matrix | Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Return variances along an axis."""
    ...


@overload
def std(arr: Vector | Matrix | Tensor) -> float:
    """Return the population standard deviation."""
    ...


@overload
def std(arr: Vector | Matrix | Tensor, axis: int, keepdims: bool = False) -> Tensor:
    """Return standard deviations along an axis."""
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


class AxisError(StrataxError):
    """Raised when an axis is out of range."""


__all__: list[str]
