from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

from ._core import full as _full
from ._core import identity as _identity
from ._core import ones as _ones
from ._core import zeros as _zeros
from .core import Shape, Tensor
from .exceptions import TypeError as StrataxTypeError


def zeros(shape: Shape | Iterable[int]) -> Tensor:
    target_shape = shape if isinstance(shape, Shape) else Shape(shape)
    return Tensor._wrap(_zeros(target_shape._impl))


def ones(shape: Shape | Iterable[int]) -> Tensor:
    target_shape = shape if isinstance(shape, Shape) else Shape(shape)
    return Tensor._wrap(_ones(target_shape._impl))


def full(shape: Shape | Iterable[int], value: float) -> Tensor:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StrataxTypeError("Tensor fill value must be a number.")

    target_shape = shape if isinstance(shape, Shape) else Shape(shape)
    return Tensor._wrap(_full(target_shape._impl, float(value)))


def identity(size: int) -> Tensor:
    if isinstance(size, bool) or not isinstance(size, int):
        raise StrataxTypeError("Identity size must be an integer.")

    return Tensor._wrap(_identity(size))