from collections.abc import Iterable, Iterator
from typing import overload


class Shape:
    """Stores non-negative dimensions for a Stratax array."""

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

    @overload
    def __init__(self, first: int, second: int, /, *rest: int) -> None:
        """Create a shape from two or more non-negative dimensions."""
        ...

    @property
    def rank(self) -> int:
        """Number of dimensions in the shape."""
        ...

    @property
    def elements(self) -> int:
        """Product of all dimensions, or zero for an empty rank-0 shape."""
        ...

    @property
    def empty(self) -> bool:
        """Whether the shape has no dimensions."""
        ...

    def __len__(self) -> int:
        """Return the rank of the shape."""
        ...

    def __getitem__(self, index: int) -> int:
        """Return one dimension by index."""
        ...

    def __iter__(self) -> Iterator[int]:
        """Iterate over dimensions in order."""
        ...

    def __eq__(self, other: object) -> bool:
        """Return whether two shapes have the same dimensions."""
        ...

    def __ne__(self, other: object) -> bool:
        """Return whether two shapes differ."""
        ...

    def __repr__(self) -> str:
        """Return a tuple-style representation of the shape."""
        ...
