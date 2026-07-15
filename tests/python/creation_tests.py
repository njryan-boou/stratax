from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from stratax import (  # noqa: E402
    Shape,
    Tensor,
    TypeError as StrataxTypeError,
    full,
    identity,
    ones,
    zeros,
)


class TestCreationInterfaceTests:
    def test_zeros_and_ones_accept_shape_and_iterable(self) -> None:
        z = zeros(Shape(2, 2))
        o = ones([2, 2])

        assert isinstance(z, Tensor)
        assert z.shape == Shape(2, 2)
        assert z.tolist() == [0.0, 0.0, 0.0, 0.0]

        assert isinstance(o, Tensor)
        assert o.shape == Shape(2, 2)
        assert o.tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_full_accepts_shape_and_iterable(self) -> None:
        a = full(Shape(1, 3), 2.5)
        b = full([2], 7.0)

        assert isinstance(a, Tensor)
        assert a.shape == Shape(1, 3)
        assert a.tolist() == [2.5, 2.5, 2.5]

        assert isinstance(b, Tensor)
        assert b.shape == Shape([2])
        assert b.tolist() == [7.0, 7.0]

    def test_identity_creates_square_tensor(self) -> None:
        eye = identity(3)

        assert isinstance(eye, Tensor)
        assert eye.shape == Shape(3, 3)
        assert eye.tolist() == [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

    def test_creation_functions_validate_input_types(self) -> None:
        with pytest.raises(StrataxTypeError):
            _ = zeros(object())

        with pytest.raises(StrataxTypeError):
            _ = ones(object())

        with pytest.raises(StrataxTypeError):
            _ = full(object(), 1.0)

        with pytest.raises(StrataxTypeError):
            _ = full([2], object())

        with pytest.raises(StrataxTypeError):
            _ = identity(object())