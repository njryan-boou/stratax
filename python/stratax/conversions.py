from __future__ import annotations

from ._core import to_matrix as _to_matrix
from ._core import to_tensor as _to_tensor
from ._core import to_vector as _to_vector
from .exceptions import TypeError as StrataxTypeError
from .matrix import Matrix
from .tensor import Tensor
from .vector import Vector


def to_vector(arr: Vector | Matrix | Tensor) -> Vector:
    if isinstance(arr, Vector):
        return Vector._wrap(_to_vector(arr._impl))
    if isinstance(arr, Matrix):
        return Vector._wrap(_to_vector(arr._impl))
    if isinstance(arr, Tensor):
        return Vector._wrap(_to_vector(arr._impl))

    raise StrataxTypeError("to_vector expects a Vector, Matrix, or Tensor.")


def to_matrix(arr: Vector | Matrix | Tensor) -> Matrix:
    if isinstance(arr, Vector):
        return Matrix._wrap(_to_matrix(arr._impl))
    if isinstance(arr, Matrix):
        return Matrix._wrap(_to_matrix(arr._impl))
    if isinstance(arr, Tensor):
        return Matrix._wrap(_to_matrix(arr._impl))

    raise StrataxTypeError("to_matrix expects a Vector, Matrix, or Tensor.")


def to_tensor(arr: Vector | Matrix | Tensor) -> Tensor:
    if isinstance(arr, Vector):
        return Tensor._wrap(_to_tensor(arr._impl))
    if isinstance(arr, Matrix):
        return Tensor._wrap(_to_tensor(arr._impl))
    if isinstance(arr, Tensor):
        return Tensor._wrap(_to_tensor(arr._impl))

    raise StrataxTypeError("to_tensor expects a Vector, Matrix, or Tensor.")