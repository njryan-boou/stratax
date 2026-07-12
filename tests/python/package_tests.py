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

    def test_public_exceptions_are_exported(self) -> None:
        assert issubclass(stratax.StrataxError, RuntimeError)
        assert issubclass(stratax.ShapeError, stratax.StrataxError)
        assert issubclass(stratax.DimensionError, stratax.StrataxError)
        assert issubclass(stratax.IndexError, stratax.StrataxError)
        assert issubclass(stratax.TypeError, stratax.StrataxError)
        assert issubclass(stratax.BroadcastError, stratax.StrataxError)
        assert issubclass(stratax.ZeroDivisionError, stratax.StrataxError)

