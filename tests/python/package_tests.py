from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import stratax


class TestPackageInterfaceTests:
    def test_public_classes_are_exported(self) -> None:
        assert stratax.__all__ == [
            "Shape",
            "Tensor",
            "Vector",
            "Matrix",
            "to_vector",
            "to_matrix",
            "to_tensor",
            "zeros",
            "ones",
            "full",
            "identity",
            "StrataxError",
            "ShapeError",
            "DimensionError",
            "IndexError",
            "TypeError",
            "BroadcastError",
            "ZeroDivisionError",
        ]
        assert stratax.Shape is stratax.shape.Shape
        assert stratax.Tensor is stratax.tensor.Tensor
        assert stratax.Vector is stratax.vector.Vector
        assert stratax.Matrix is stratax.matrix.Matrix
        assert stratax.to_vector is stratax.conversions.to_vector
        assert stratax.to_matrix is stratax.conversions.to_matrix
        assert stratax.to_tensor is stratax.conversions.to_tensor
        assert stratax.zeros is stratax.creation.zeros
        assert stratax.ones is stratax.creation.ones
        assert stratax.full is stratax.creation.full
        assert stratax.identity is stratax.creation.identity

    def test_public_exceptions_are_exported(self) -> None:
        assert issubclass(stratax.StrataxError, RuntimeError)
        assert issubclass(stratax.ShapeError, stratax.StrataxError)
        assert issubclass(stratax.DimensionError, stratax.StrataxError)
        assert issubclass(stratax.IndexError, stratax.StrataxError)
        assert issubclass(stratax.TypeError, stratax.StrataxError)
        assert issubclass(stratax.BroadcastError, stratax.StrataxError)
        assert issubclass(stratax.ZeroDivisionError, stratax.StrataxError)

