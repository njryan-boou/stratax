from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = next(candidate for candidate in Path(__file__).resolve().parents if (candidate / "python" / "stratax").exists())
sys.path.insert(0, str(ROOT / "python"))

from stratax import (  # noqa: E402
    Matrix,
    Shape,
    Tensor,
    TypeError as StrataxTypeError,
    Vector,
    full,
    identity,
    ones,
    zeros,
)


class TestCreationInterfaceTests:
    def test_zeros_and_ones_accept_shape(self) -> None:
        z = zeros(Shape([2, 2]))
        o = ones(Shape([2, 2]))

        assert isinstance(z, Tensor)
        assert z.shape == Shape([2, 2])
        assert z.tolist() == [[0.0, 0.0], [0.0, 0.0]]

        assert isinstance(o, Tensor)
        assert o.shape == Shape([2, 2])
        assert o.tolist() == [[1.0, 1.0], [1.0, 1.0]]

    def test_full_accepts_shape(self) -> None:
        a = full(Shape([1, 3]), 2.5)
        b = full(Shape([2]), 7.0)

        assert isinstance(a, Tensor)
        assert a.shape == Shape([1, 3])
        assert a.tolist() == [[2.5, 2.5, 2.5]]

        assert isinstance(b, Tensor)
        assert b.shape == Shape([2])
        assert b.tolist() == [7.0, 7.0]

    def test_zeros_and_ones_accept_vector_size(self) -> None:
        z = zeros(3)
        o = ones(4)

        assert isinstance(z, Vector)
        assert z.shape == Shape([3])
        assert z.tolist() == [0.0, 0.0, 0.0]

        assert isinstance(o, Vector)
        assert o.shape == Shape([4])
        assert o.tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_zeros_and_ones_accept_matrix_dimensions(self) -> None:
        z = zeros(2, 3)
        o = ones(2, 2)

        assert isinstance(z, Matrix)
        assert z.shape == Shape([2, 3])
        assert z.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        assert isinstance(o, Matrix)
        assert o.shape == Shape([2, 2])
        assert o.tolist() == [[1.0, 1.0], [1.0, 1.0]]

    def test_full_accepts_vector_and_matrix_dimensions(self) -> None:
        vector = full(3, 2.5)
        matrix = full(2, 3, -1.5)

        assert isinstance(vector, Vector)
        assert vector.shape == Shape([3])
        assert vector.tolist() == [2.5, 2.5, 2.5]

        assert isinstance(matrix, Matrix)
        assert matrix.shape == Shape([2, 3])
        assert matrix.tolist() == [[-1.5, -1.5, -1.5], [-1.5, -1.5, -1.5]]

    def test_identity_creates_square_matrix(self) -> None:
        eye = identity(3)

        assert isinstance(eye, Matrix)
        assert eye.shape == Shape([3, 3])
        assert eye.tolist() == [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    def test_creation_functions_validate_input_types(self) -> None:
        with pytest.raises(TypeError):
            _ = zeros(object())

        with pytest.raises(TypeError):
            _ = ones(object())

        with pytest.raises(TypeError):
            _ = full(object(), 1.0)

        with pytest.raises(TypeError):
            _ = full(Shape([2]), object())

        with pytest.raises(TypeError):
            _ = identity(object())
