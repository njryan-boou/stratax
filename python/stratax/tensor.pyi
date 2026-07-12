from collections.abc import Iterable, Iterator
from typing import overload

from .shape import Shape


class Tensor:
    """N-dimensional row-major floating-point array with non-negative dimensions."""

    @overload
    def __init__(self) -> None:
        """Create an empty rank-0 tensor."""
        ...

    @overload
    def __init__(self, other: "Tensor") -> None:
        """Copy another tensor."""
        ...

    @overload
    def __init__(self, shape: Shape) -> None:
        """Create a tensor from a shape."""
        ...

    @overload
    def __init__(self, shape: Shape, value: float) -> None:
        """Create a tensor and fill every element with value."""
        ...

    @overload
    def __init__(self, shape: Iterable[int]) -> None:
        """Create a tensor from non-negative integer dimensions."""
        ...

    @overload
    def __init__(self, shape: Iterable[int], value: float) -> None:
        """Create a tensor from dimensions and fill it with value."""
        ...

    @property
    def size(self) -> int:
        """Total number of elements."""
        ...

    @property
    def rank(self) -> int:
        """Number of tensor dimensions."""
        ...

    @property
    def empty(self) -> bool:
        """Whether the tensor contains no elements."""
        ...

    @property
    def shape(self) -> Shape:
        """Shape describing all tensor dimensions."""
        ...

    @property
    def strides(self) -> list[int]:
        """Row-major stride values."""
        ...

    def fill(self, value: float) -> None:
        """Set every element to value."""
        ...

    def tolist(self) -> list[float]:
        """Return tensor data as a flat Python list."""
        ...

    def __len__(self) -> int:
        """Return the number of flat elements."""
        ...

    def __getitem__(self, index: int | tuple[int, ...]) -> float:
        """Return an element by non-negative flat index or N-dimensional tuple."""
        ...

    def __setitem__(self, index: int | tuple[int, ...], value: float) -> None:
        """Set an element by non-negative flat index or N-dimensional tuple."""
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
        """Return this tensor with every element negated."""
        ...

    def __repr__(self) -> str:
        """Return a compact tensor representation."""
        ...
