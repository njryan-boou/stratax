from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

from ._core import Matrix as _Matrix
from .exceptions import TypeError as StrataxTypeError
from .shape import Shape


class Matrix:
    def __init__(self, *args):
        if not args:
            self._impl = _Matrix()

        elif len(args) == 1:
            value = args[0]

            if isinstance(value, Matrix):
                self._impl = _Matrix(value._impl)

            elif isinstance(value, _Matrix):
                self._impl = _Matrix(value)

            elif isinstance(value, Shape):
                self._impl = _Matrix(value._impl)

            elif isinstance(value, Iterable):
                self._impl = _Matrix(value)

            else:
                raise StrataxTypeError(
                    "Matrix constructor expects a Matrix, Shape, or iterable of row values."
                )

        elif len(args) == 2:
            rows, cols = args
            if not isinstance(rows, int) or not isinstance(cols, int):
                raise StrataxTypeError("Matrix row and column counts must be integers.")

            self._impl = _Matrix(rows, cols)

        elif len(args) == 3:
            rows, cols, value = args
            if not isinstance(rows, int) or not isinstance(cols, int):
                raise StrataxTypeError("Matrix row and column counts must be integers.")

            self._impl = _Matrix(rows, cols, value)

        else:
            raise TypeError("Matrix accepts at most three arguments.")

    @property
    def size(self):
        return self._impl.size()

    @property
    def rank(self):
        return self._impl.rank()

    @property
    def empty(self):
        return self._impl.empty()

    @property
    def rows(self):
        return self._impl.rows()

    @property
    def cols(self):
        return self._impl.cols()

    @property
    def shape(self):
        return Shape(self._impl.shape())

    @property
    def strides(self):
        return self._impl.strides()

    def fill(self, value):
        self._impl.fill(value)

    def tolist(self):
        return self._impl.tolist()

    def reshape(self, shape):
        from .tensor import Tensor

        target_shape = shape if isinstance(shape, Shape) else Shape(shape)
        return Tensor._wrap(self._impl.reshape(target_shape._impl))

    def flatten(self):
        from .vector import Vector

        return Vector._wrap(self._impl.flatten())

    @staticmethod
    def _wrap(impl):
        matrix = Matrix.__new__(Matrix)
        matrix._impl = impl
        return matrix

    @staticmethod
    def _operand(value):
        if isinstance(value, Matrix):
            return value._impl

        if isinstance(value, Real):
            return float(value)

        return NotImplemented

    def __len__(self):
        return len(self._impl)

    def __getitem__(self, index):
        value = self._impl[index]
        if isinstance(value, _Matrix):
            return self._wrap(value)

        return value

    def __setitem__(self, index, value):
        self._impl[index] = value

    def __iter__(self):
        return iter(self._impl)

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented

        return self._impl == other._impl

    def __ne__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented

        return self._impl != other._impl

    def __add__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        return self._wrap(self._impl + operand)

    def __radd__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented or isinstance(other, Matrix):
            return NotImplemented

        return self._wrap(operand + self._impl)

    def __sub__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        return self._wrap(self._impl - operand)

    def __rsub__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented or isinstance(other, Matrix):
            return NotImplemented

        return self._wrap(operand - self._impl)

    def __mul__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        return self._wrap(self._impl * operand)

    def __rmul__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented or isinstance(other, Matrix):
            return NotImplemented

        return self._wrap(operand * self._impl)

    def __truediv__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        return self._wrap(self._impl / operand)

    def __rtruediv__(self, other):
        operand = self._operand(other)
        if operand is NotImplemented or isinstance(other, Matrix):
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

    def __repr__(self):
        return repr(self._impl)