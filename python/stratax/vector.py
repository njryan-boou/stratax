from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

from ._core import Vector as _Vector
from .exceptions import TypeError as StrataxTypeError

from .shape import Shape


class Vector:
    def __init__(self, *args):
        if not args:
            self._impl = _Vector()

        elif len(args) == 1:
            value = args[0]

            if isinstance(value, Vector):
                self._impl = _Vector(value._impl)

            elif isinstance(value, _Vector):
                self._impl = _Vector(value)

            elif isinstance(value, Shape):
                self._impl = _Vector(value._impl)

            elif isinstance(value, int):
                self._impl = _Vector(value)

            elif isinstance(value, Iterable):
                self._impl = _Vector(value)

            else:
                raise StrataxTypeError(
                    "Vector constructor expects a Vector, Shape, size, or iterable of numbers."
                )

        elif len(args) == 2:
            size, value = args
            if not isinstance(size, int):
                raise StrataxTypeError("Vector size must be an integer.")

            self._impl = _Vector(size, value)

        else:
            raise TypeError("Vector accepts at most two arguments.")

    @property
    def size(self) -> int:
        return self._impl.size()

    @property
    def rank(self) -> int:
        return self._impl.rank()

    @property
    def empty(self) -> bool:
        return self._impl.empty()

    @property
    def shape(self) -> Shape:
        return Shape(self._impl.shape())

    @property
    def strides(self) -> list[int]:
        return self._impl.strides()

    def fill(self, value: float) -> None:
        self._impl.fill(value)

    def tolist(self) -> list[float]:
        return self._impl.tolist()

    @staticmethod
    def _wrap(impl: _Vector) -> "Vector":
        vector = Vector.__new__(Vector)
        vector._impl = impl
        return vector

    @staticmethod
    def _operand(value):
        if isinstance(value, Vector):
            return value._impl

        if isinstance(value, Real):
            return float(value)

        return NotImplemented

    def __len__(self) -> int:
        return len(self._impl)

    def __getitem__(self, index: int) -> float:
        return self._impl[index]

    def __setitem__(self, index: int, value: float) -> None:
        self._impl[index] = value

    def __iter__(self):
        return iter(self._impl)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented

        return self._impl == other._impl

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented

        return self._impl != other._impl

    def __add__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        return self._wrap(self._impl + operand)

    def __radd__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented or isinstance(other, Vector):
            return NotImplemented

        return self._wrap(operand + self._impl)

    def __sub__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        return self._wrap(self._impl - operand)

    def __rsub__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented or isinstance(other, Vector):
            return NotImplemented

        return self._wrap(operand - self._impl)

    def __mul__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        return self._wrap(self._impl * operand)

    def __rmul__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented or isinstance(other, Vector):
            return NotImplemented

        return self._wrap(operand * self._impl)

    def __truediv__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        return self._wrap(self._impl / operand)

    def __rtruediv__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented or isinstance(other, Vector):
            return NotImplemented

        return self._wrap(operand / self._impl)

    def __iadd__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        self._impl += operand
        return self

    def __isub__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        self._impl -= operand
        return self

    def __imul__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        self._impl *= operand
        return self

    def __itruediv__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        self._impl /= operand
        return self

    def __pos__(self):
        return self._wrap(+self._impl)

    def __neg__(self):
        return self._wrap(-self._impl)

    def __repr__(self) -> str:
        return repr(self._impl)
