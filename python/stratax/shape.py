from __future__ import annotations

from collections.abc import Iterable

from ._core import _Shape
from .exceptions import TypeError as StrataxTypeError



class Shape:
    """Stores the dimensions of a Stratax array."""

    def __init__(self, *args):
        """Create an empty shape, copy a shape, or build one from dimensions."""
        if not args:
            self._impl = _Shape()

        elif len(args) == 1:
            value = args[0]

            if (
                isinstance(value, _Shape)
                or not isinstance(value, Shape)
                and isinstance(value, Iterable)
            ):
                self._impl = _Shape(value)

            elif isinstance(value, Shape):
                self._impl = _Shape(value._impl)

            else:
                raise StrataxTypeError(
                    "Shape constructor expects a Shape or iterable of integers."
                )

        else:
            self._impl = _Shape(args)

    @property
    def rank(self) -> int:
        """Number of dimensions in the shape."""
        return self._impl.rank()

    @property
    def elements(self) -> int:
        """Total element count described by the dimensions."""
        return self._impl.elements()

    @property
    def empty(self) -> bool:
        """Whether the shape has no dimensions."""
        return self._impl.empty()

    def __len__(self) -> int:
        """Return the rank of the shape."""
        return len(self._impl)

    def __getitem__(self, index: int) -> int:
        """Return one dimension by index."""
        return self._impl[index]

    def __iter__(self):
        """Iterate over dimensions in order."""
        return iter(self._impl)

    def __eq__(self, other: object) -> bool:
        """Return whether two shapes have the same dimensions."""
        if not isinstance(other, Shape):
            return NotImplemented

        return self._impl == other._impl

    def __ne__(self, other: object) -> bool:
        """Return whether two shapes differ."""
        if not isinstance(other, Shape):
            return NotImplemented

        return self._impl != other._impl

    def __repr__(self) -> str:
        """Return a tuple-style representation of the shape."""
        return repr(self._impl)
