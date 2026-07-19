from __future__ import annotations

from collections.abc import Iterable

from ._core import _Shape
from .exceptions import TypeError as StrataxTypeError


class Shape:
    def __init__(self, *args):
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
    def rank(self):
        return self._impl.rank()

    @property
    def elements(self):
        return self._impl.elements()

    @property
    def empty(self):
        return self._impl.empty()

    def __len__(self):
        return len(self._impl)

    def __getitem__(self, index):
        return self._impl[index]

    def __iter__(self):
        return iter(self._impl)

    def __eq__(self, other):
        if not isinstance(other, Shape):
            return NotImplemented

        return self._impl == other._impl

    def __ne__(self, other):
        if not isinstance(other, Shape):
            return NotImplemented

        return self._impl != other._impl

    def __repr__(self):
        return repr(self._impl)
