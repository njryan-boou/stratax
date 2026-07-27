from collections.abc import Iterable
from numbers import Real
from operator import add, iadd, imul, isub, itruediv, mul, sub, truediv

from ._core import Tensor as _Tensor
from .exceptions import TypeError as StrataxTypeError
from .shape import Shape


class Tensor:
    def __init__(self, *args):
        if not args:
            self._impl = _Tensor()

        elif len(args) == 1:
            value = args[0]

            if isinstance(value, Tensor):
                self._impl = _Tensor(value._impl)

            elif isinstance(value, _Tensor):
                self._impl = _Tensor(value)

            elif isinstance(value, Shape):
                self._impl = _Tensor(value._impl)

            elif isinstance(value, Iterable):
                self._impl = _Tensor(value)

            else:
                raise StrataxTypeError(
                    "Tensor constructor expects a Tensor, Shape, or iterable of dimensions."
                )

        elif len(args) == 2:
            shape, value = args

            if isinstance(shape, Shape):
                self._impl = _Tensor(shape._impl, value)

            elif isinstance(shape, Iterable):
                self._impl = _Tensor(shape, value)

            else:
                raise StrataxTypeError("Tensor shape must be a Shape or iterable of dimensions.")

        else:
            raise TypeError("Tensor accepts at most two arguments.")

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
        target_shape = shape if isinstance(shape, Shape) else Shape(shape)
        return self._wrap(self._impl.reshape(target_shape._impl))

    def flatten(self):
        from .vector import Vector

        return Vector._wrap(self._impl.flatten())

    @staticmethod
    def _wrap(impl):
        tensor = Tensor.__new__(Tensor)
        tensor._impl = impl
        return tensor

    @staticmethod
    def _operand(value):
        if isinstance(value, Tensor):
            return value._impl

        if isinstance(value, Real):
            return float(value)

        return NotImplemented

    def __len__(self):
        return len(self._impl)

    def __getitem__(self, index):
        value = self._impl[index]
        if isinstance(value, _Tensor):
            return self._wrap(value)

        return value

    def __setitem__(self, index, value):
        self._impl[index] = value

    def __iter__(self):
        return iter(self._impl)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented

        return self._impl == other._impl

    def __ne__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented

        return self._impl != other._impl

    def _binary_op(self, other, operator, reverse=False):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        if reverse and isinstance(other, Tensor):
            return NotImplemented

        if reverse:
            return self._wrap(operator(operand, self._impl))

        return self._wrap(operator(self._impl, operand))

    def _inplace_op(self, other, operator):
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented

        operator(self._impl, operand)
        return self

    def __add__(self, other):
        return self._binary_op(other, add)

    def __radd__(self, other):
        return self._binary_op(other, add, reverse=True)

    def __sub__(self, other):
        return self._binary_op(other, sub)

    def __rsub__(self, other):
        return self._binary_op(other, sub, reverse=True)

    def __mul__(self, other):
        return self._binary_op(other, mul)

    def __rmul__(self, other):
        return self._binary_op(other, mul, reverse=True)

    def __truediv__(self, other):
        return self._binary_op(other, truediv)

    def __rtruediv__(self, other):
        return self._binary_op(other, truediv, reverse=True)

    def __iadd__(self, other):
        return self._inplace_op(other, iadd)

    def __isub__(self, other):
        return self._inplace_op(other, isub)

    def __imul__(self, other):
        return self._inplace_op(other, imul)

    def __itruediv__(self, other):
        return self._inplace_op(other, itruediv)

    def __pos__(self):
        return self._wrap(+self._impl)

    def __neg__(self):
        return self._wrap(-self._impl)

    def __repr__(self):
        return repr(self._impl)


__all__ = ["Tensor"]