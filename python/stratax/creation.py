from collections.abc import Iterable
from numbers import Real

from ._core import full as _full
from ._core import identity as _identity
from ._core import ones as _ones
from ._core import zeros as _zeros
from .shape import Shape
from .tensor import Tensor
from .exceptions import TypeError as StrataxTypeError


def _as_shape(shape):
    return shape if isinstance(shape, Shape) else Shape(shape)


def _wrap_creation(fn, shape):
    target_shape = _as_shape(shape)
    return Tensor._wrap(fn(target_shape._impl))


def _make_unary_creation(fn):
    def _creator(shape):
        return _wrap_creation(fn, shape)

    return _creator


zeros = _make_unary_creation(_zeros)
ones = _make_unary_creation(_ones)


def full(shape, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StrataxTypeError("Tensor fill value must be a number.")

    target_shape = _as_shape(shape)
    return Tensor._wrap(_full(target_shape._impl, float(value)))


def identity(size):
    if isinstance(size, bool) or not isinstance(size, int):
        raise StrataxTypeError("Identity size must be an integer.")

    return Tensor._wrap(_identity(size))


__all__ = ["zeros", "ones", "full", "identity"]