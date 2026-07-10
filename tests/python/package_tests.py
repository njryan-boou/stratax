from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import stratax


class PackageInterfaceTests(unittest.TestCase):
    def test_public_classes_are_exported(self) -> None:
        self.assertEqual(
            stratax.__all__,
            [
                "Shape",
                "Tensor",
                "Vector",
                "Matrix",
                "StrataxError",
                "ShapeError",
                "DimensionError",
                "IndexError",
                "DTypeError",
                "BroadcastError",
                "ZeroDivisionError",
            ],
        )
        self.assertIs(stratax.Shape, stratax.shape.Shape)
        self.assertIs(stratax.Tensor, stratax.tensor.Tensor)
        self.assertIs(stratax.Vector, stratax.vector.Vector)
        self.assertIs(stratax.Matrix, stratax.matrix.Matrix)

    def test_public_exceptions_are_exported(self) -> None:
        self.assertTrue(issubclass(stratax.StrataxError, RuntimeError))
        self.assertTrue(issubclass(stratax.ShapeError, stratax.StrataxError))
        self.assertTrue(issubclass(stratax.DimensionError, stratax.StrataxError))
        self.assertTrue(issubclass(stratax.IndexError, stratax.StrataxError))
        self.assertTrue(issubclass(stratax.DTypeError, stratax.StrataxError))
        self.assertTrue(issubclass(stratax.BroadcastError, stratax.StrataxError))
        self.assertTrue(issubclass(stratax.ZeroDivisionError, stratax.StrataxError))


if __name__ == "__main__":
    unittest.main()
