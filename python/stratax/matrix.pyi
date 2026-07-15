from collections.abc import Iterable, Iterator
from typing import overload

from .shape import Shape
from .tensor import Tensor
from .vector import Vector


class Matrix:
    """Rank-2 row-major floating-point array with non-negative dimensions."""

    @overload
    def __init__(self) -> None:
        """Create an empty 0 by 0 matrix."""
        ...

    @overload
    def __init__(self, other: "Matrix") -> None:
        """Copy another matrix."""
        ...

    @overload
    def __init__(self, shape: Shape) -> None:
        """Create a matrix from a rank-2 shape."""
        ...

    @overload
    def __init__(self, rows: int, cols: int) -> None:
        """Create a matrix with default-initialized elements."""
        ...

    @overload
    def __init__(self, rows: int, cols: int, value: float) -> None:
        """Create a matrix and fill every element with value."""
        ...

    @overload
    def __init__(self, values: Iterable[Iterable[float]]) -> None:
        """Create a matrix from equal-length rows of numeric values."""
        ...

    @property
    def size(self) -> int:
        """Total number of elements."""
        ...

    @property
    def rank(self) -> int:
        """Rank of the matrix, always two."""
        ...

    @property
    def empty(self) -> bool:
        """Whether the matrix contains no elements."""
        ...

    @property
    def rows(self) -> int:
        """Number of matrix rows."""
        ...

    @property
    def cols(self) -> int:
        """Number of matrix columns."""
        ...

    @property
    def shape(self) -> Shape:
        """Shape describing rows and columns."""
        ...

    @property
    def strides(self) -> list[int]:
        """Row-major stride values."""
        ...

    def fill(self, value: float) -> None:
        """Set every element to value."""
        ...

    def tolist(self) -> list[list[float]]:
        """Return matrix data as nested Python lists."""
        ...

    def reshape(self, shape: Shape | Iterable[int]) -> Tensor:
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
        """Return an element by non-negative (row, col)."""
        ...

    @overload
    def __getitem__(self, index: slice | tuple[int | slice, int | slice]) -> "Matrix":
        """Return a matrix slice using Python slice syntax."""
        ...

    def __setitem__(self, index: tuple[int, int], value: float) -> None:
        """Set an element by non-negative (row, col)."""
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
        """Return this matrix with every element negated."""
        ...

    def __repr__(self) -> str:
        """Return a compact matrix representation."""
        ...
