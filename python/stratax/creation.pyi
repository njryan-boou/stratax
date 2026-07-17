from collections.abc import Iterable

from .core import Shape, Tensor


def zeros(shape: Shape | Iterable[int]) -> Tensor:
    ...


def ones(shape: Shape | Iterable[int]) -> Tensor:
    ...


def full(shape: Shape | Iterable[int], value: float) -> Tensor:
    ...

def identity(size: int) -> Tensor:
    ...