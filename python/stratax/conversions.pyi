from .core import Matrix, Tensor, Vector


def to_vector(arr: Vector | Matrix | Tensor) -> Vector:
    ...


def to_matrix(arr: Vector | Matrix | Tensor) -> Matrix:
    ...


def to_tensor(arr: Vector | Matrix | Tensor) -> Tensor:
    ...