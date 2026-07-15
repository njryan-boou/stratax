from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from stratax import (  # noqa: E402
    Matrix,
    Shape,
    ShapeError,
    Tensor,
    TypeError as StrataxTypeError,
    Vector,
    to_matrix,
    to_tensor,
    to_vector,
)


class TestConversionsInterfaceTests:
    def test_to_tensor_from_vector_and_matrix(self) -> None:
        vector = Vector([1.0, 2.0, 3.0])
        matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])

        as_tensor_from_vector = to_tensor(vector)
        as_tensor_from_matrix = to_tensor(matrix)

        assert isinstance(as_tensor_from_vector, Tensor)
        assert as_tensor_from_vector.shape == Shape([3])
        assert as_tensor_from_vector.tolist() == [1.0, 2.0, 3.0]

        assert isinstance(as_tensor_from_matrix, Tensor)
        assert as_tensor_from_matrix.shape == Shape(2, 2)
        assert as_tensor_from_matrix.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_to_vector_from_rank_one_tensor(self) -> None:
        tensor = Tensor([3])
        tensor[0] = 2.0
        tensor[1] = 4.0
        tensor[2] = 6.0

        vector = to_vector(tensor)

        assert isinstance(vector, Vector)
        assert vector.shape == Shape([3])
        assert vector.tolist() == [2.0, 4.0, 6.0]

    def test_to_vector_rejects_non_rank_one(self) -> None:
        matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ShapeError):
            _ = to_vector(matrix)

    def test_to_matrix_from_rank_two_tensor(self) -> None:
        tensor = Tensor([2, 2])
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0]):
            tensor[index] = value

        matrix = to_matrix(tensor)

        assert isinstance(matrix, Matrix)
        assert matrix.shape == Shape(2, 2)
        assert matrix.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_to_matrix_from_matrix_shaped_tensor_with_singletons(self) -> None:
        tensor = Tensor([1, 2, 1, 3])
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            tensor[index] = value

        matrix = to_matrix(tensor)

        assert isinstance(matrix, Matrix)
        assert matrix.shape == Shape(2, 3)
        assert matrix.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_to_matrix_rejects_non_rank_two(self) -> None:
        vector = Vector([1.0, 2.0])

        with pytest.raises(ShapeError):
            _ = to_matrix(vector)

    def test_to_matrix_rejects_non_matrix_shape(self) -> None:
        tensor = Tensor([3])

        with pytest.raises(ShapeError):
            _ = to_matrix(tensor)

    def test_conversion_functions_validate_argument_type(self) -> None:
        with pytest.raises(StrataxTypeError):
            _ = to_vector(object())

        with pytest.raises(StrataxTypeError):
            _ = to_matrix(object())

        with pytest.raises(StrataxTypeError):
            _ = to_tensor(object())