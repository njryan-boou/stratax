from ._core import to_matrix as _to_matrix
from ._core import to_tensor as _to_tensor
from ._core import to_vector as _to_vector
from .matrix import Matrix
from .tensor import Tensor
from .vector import Vector
from .exceptions import TypeError as StrataxTypeError


def _unwrap_array(arr, name):
    if isinstance(arr, (Vector, Matrix, Tensor)):
        return arr._impl

    raise StrataxTypeError(f"{name} expects a Vector, Matrix, or Tensor.")


def _make_converter(core_fn, wrapper, name):
    def _converter(arr):
        return wrapper._wrap(core_fn(_unwrap_array(arr, name)))

    _converter.__name__ = name
    return _converter


to_vector = _make_converter(_to_vector, Vector, "to_vector")
to_matrix = _make_converter(_to_matrix, Matrix, "to_matrix")
to_tensor = _make_converter(_to_tensor, Tensor, "to_tensor")


__all__ = ["to_vector", "to_matrix", "to_tensor"]