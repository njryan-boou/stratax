from collections.abc import Iterable, Iterator
from typing import overload

from .shape import Shape
from .tensor import Tensor


class Vector:
    """Rank-1 floating-point array with non-negative sizes."""

    @overload
    def __init__(self) -> None:
        """Create an empty vector."""
        ...

    @overload
    def __init__(self, other: "Vector") -> None:
        """Copy another vector."""
        ...

    @overload
    def __init__(self, shape: Shape) -> None:
        """Create a vector from a rank-1 shape."""
        ...

    @overload
    def __init__(self, size: int) -> None:
        """Create a vector with size default-initialized elements."""
        ...

    @overload
    def __init__(self, size: int, value: float) -> None:
        """Create a vector and fill every element with value."""
        ...

    @overload
    def __init__(self, values: Iterable[float]) -> None:
        """Create a vector from numeric values."""
        ...

    @property
    def size(self) -> int:
        """Number of elements in the vector."""
        ...

    @property
    def rank(self) -> int:
        """Rank of the vector, always one."""
        ...

    @property
    def empty(self) -> bool:
        """Whether the vector contains no elements."""
        ...

    @property
    def shape(self) -> Shape:
        """Shape describing the vector length."""
        ...

    @property
    def strides(self) -> list[int]:
        """Row-major stride values for the vector."""
        ...

    def fill(self, value: float) -> None:
        """Set every element to value."""
        ...

    def tolist(self) -> list[float]:
        """Return the vector data as a Python list."""
        ...

    def reshape(self, shape: Shape | Iterable[int]) -> Tensor:
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
        """Return this vector with every element negated."""
        ...

    def __repr__(self) -> str:
        """Return a compact vector representation."""
        ...
