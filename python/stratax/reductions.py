from . import _core
from .exceptions import TypeError as StrataxTypeError
from .matrix import Matrix
from .tensor import Tensor
from .vector import Vector


def _unwrap_array(arr):
    if isinstance(arr, (Vector, Matrix, Tensor)):
        return arr._impl

    if isinstance(arr, (_core.Vector, _core.Matrix, _core.Tensor)):
        return arr

    raise StrataxTypeError("reduction expects a Vector, Matrix, or Tensor.")


def _wrap_result(value):
    if isinstance(value, _core.Tensor):
        return Tensor._wrap(value)

    return value


def _call_global(name, arr):
    impl = _unwrap_array(arr)
    return getattr(_core, name)(impl)


def _call_axis(name, arr, axis, keepdims=False):
    impl = _unwrap_array(arr)

    if isinstance(axis, bool) or not isinstance(axis, int):
        raise StrataxTypeError("axis must be an integer.")

    return _wrap_result(getattr(_core, name)(impl, axis, keepdims))


def _make_reduction(name):
    def _reduction(arr, axis=None, keepdims=False):
        if axis is None:
            return _call_global(name, arr)

        return _call_axis(name, arr, axis, keepdims)

    _reduction.__name__ = name
    return _reduction


_REDUCTION_NAMES = ["sum", "prod", "max", "min", "argmax", "argmin", "mean", "var", "std"]

for _name in _REDUCTION_NAMES:
    globals()[_name] = _make_reduction(_name)

del _name

__all__ = _REDUCTION_NAMES
