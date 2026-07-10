from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from stratax import DTypeError, DimensionError, IndexError as StrataxIndexError
from stratax import Shape, Vector, ZeroDivisionError as StrataxZeroDivisionError


class VectorInterfaceTests(unittest.TestCase):
    def test_default_vector_is_empty_rank_one_vector(self) -> None:
        vector = Vector()

        self.assertEqual(vector.size, 0)
        self.assertEqual(vector.rank, 1)
        self.assertTrue(vector.empty)
        self.assertEqual(len(vector), 0)
        self.assertEqual(vector.shape, Shape([0]))
        self.assertEqual(vector.strides, [1])
        self.assertEqual(vector.tolist(), [])
        self.assertEqual(list(vector), [])

    def test_size_constructor_builds_default_values(self) -> None:
        vector = Vector(3)

        self.assertEqual(vector.size, 3)
        self.assertEqual(vector.rank, 1)
        self.assertFalse(vector.empty)
        self.assertEqual(vector.shape, Shape([3]))
        self.assertEqual(vector.strides, [1])
        self.assertEqual(len(vector), 3)
        self.assertEqual(vector.tolist(), [0.0, 0.0, 0.0])

    def test_size_value_constructor_fills_values(self) -> None:
        vector = Vector(4, 2.5)

        self.assertEqual(vector.size, 4)
        self.assertEqual(vector.tolist(), [2.5, 2.5, 2.5, 2.5])

    def test_shape_constructor_builds_vector(self) -> None:
        vector = Vector(Shape([3]))

        self.assertEqual(vector.size, 3)
        self.assertEqual(vector.shape, Shape([3]))
        self.assertEqual(vector.tolist(), [0.0, 0.0, 0.0])

    def test_iterable_constructor_preserves_values(self) -> None:
        vector = Vector([1.0, 2.5, 3.0])

        self.assertEqual(vector.size, 3)
        self.assertEqual(vector.shape, Shape([3]))
        self.assertEqual(vector.tolist(), [1.0, 2.5, 3.0])
        self.assertEqual(list(vector), [1.0, 2.5, 3.0])

    def test_generator_constructor_preserves_values(self) -> None:
        vector = Vector(float(value) for value in (1, 2, 3))

        self.assertEqual(vector.tolist(), [1.0, 2.0, 3.0])

    def test_copy_constructor_preserves_values(self) -> None:
        original = Vector([1.0, 2.0, 3.0])
        copied = Vector(original)

        self.assertIsNot(copied, original)
        self.assertEqual(copied.tolist(), original.tolist())
        copied[0] = 9.0
        self.assertEqual(original.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(copied.tolist(), [9.0, 2.0, 3.0])

    def test_indexing_reads_and_writes_values(self) -> None:
        vector = Vector([1.0, 2.0, 3.0])

        self.assertEqual(vector[1], 2.0)
        vector[1] = 8.0
        self.assertEqual(vector[1], 8.0)
        self.assertEqual(vector.tolist(), [1.0, 8.0, 3.0])

    def test_fill_updates_all_values(self) -> None:
        vector = Vector([1.0, 2.0, 3.0])

        vector.fill(4.5)

        self.assertEqual(vector.tolist(), [4.5, 4.5, 4.5])

    def test_repr_returns_vector_text(self) -> None:
        self.assertEqual(repr(Vector([1.0, 2.0, 3.0])), "[1, 2, 3]")

    def test_equality_and_inequality(self) -> None:
        self.assertEqual(Vector([1.0, 2.0, 3.0]), Vector([1.0, 2.0, 3.0]))
        self.assertNotEqual(Vector([1.0, 2.0, 3.0]), Vector([1.0, 9.0, 3.0]))
        self.assertNotEqual(Vector([1.0, 2.0, 3.0]), Vector([1.0, 2.0]))
        self.assertNotEqual(Vector([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])

    def test_array_arithmetic(self) -> None:
        lhs = Vector([8.0, 12.0, 20.0])
        rhs = Vector([2.0, 3.0, 5.0])

        self.assertEqual((lhs + rhs).tolist(), [10.0, 15.0, 25.0])
        self.assertEqual((lhs - rhs).tolist(), [6.0, 9.0, 15.0])
        self.assertEqual((lhs * rhs).tolist(), [16.0, 36.0, 100.0])
        self.assertEqual((lhs / rhs).tolist(), [4.0, 4.0, 4.0])

        with self.assertRaises(StrataxZeroDivisionError):
            _ = lhs / Vector([1.0, 0.0, 1.0])

    def test_scalar_arithmetic(self) -> None:
        vector = Vector([2.0, 4.0, 8.0])

        self.assertEqual((vector + 2).tolist(), [4.0, 6.0, 10.0])
        self.assertEqual((vector - 2).tolist(), [0.0, 2.0, 6.0])
        self.assertEqual((vector * 2).tolist(), [4.0, 8.0, 16.0])
        self.assertEqual((vector / 2).tolist(), [1.0, 2.0, 4.0])

        with self.assertRaises(StrataxZeroDivisionError):
            _ = vector / 0

    def test_reverse_scalar_arithmetic(self) -> None:
        vector = Vector([2.0, 4.0, 8.0])

        self.assertEqual((2 + vector).tolist(), [4.0, 6.0, 10.0])
        self.assertEqual((20 - vector).tolist(), [18.0, 16.0, 12.0])
        self.assertEqual((2 * vector).tolist(), [4.0, 8.0, 16.0])
        self.assertEqual((32 / vector).tolist(), [16.0, 8.0, 4.0])

    def test_in_place_arithmetic(self) -> None:
        vector = Vector([2.0, 4.0, 8.0])

        vector += 2
        self.assertEqual(vector.tolist(), [4.0, 6.0, 10.0])

        vector -= Vector([1.0, 2.0, 3.0])
        self.assertEqual(vector.tolist(), [3.0, 4.0, 7.0])

        vector *= 2
        self.assertEqual(vector.tolist(), [6.0, 8.0, 14.0])

        vector /= Vector([3.0, 4.0, 7.0])
        self.assertEqual(vector.tolist(), [2.0, 2.0, 2.0])

    def test_unary_arithmetic(self) -> None:
        vector = Vector([1.0, -2.0, 3.0])

        self.assertEqual((+vector).tolist(), [1.0, -2.0, 3.0])
        self.assertEqual((-vector).tolist(), [-1.0, 2.0, -3.0])

    def test_invalid_constructor_argument_raises_type_error(self) -> None:
        with self.assertRaises(DTypeError):
            Vector(object())

    def test_too_many_constructor_arguments_raise_type_error(self) -> None:
        with self.assertRaises(TypeError):
            Vector(1, 2, 3)

    def test_size_value_constructor_requires_integer_size(self) -> None:
        with self.assertRaises(DTypeError):
            Vector(1.5, 2.0)

    def test_unsupported_arithmetic_operand_raises_type_error(self) -> None:
        vector = Vector([1.0, 2.0])

        with self.assertRaises(TypeError):
            _ = vector + object()

        with self.assertRaises(TypeError):
            _ = object() + vector

    def test_negative_size_raises_error(self) -> None:
        with self.assertRaises(DimensionError):
            Vector(-1)

    def test_size_overflow_raises_overflow_error(self) -> None:
        with self.assertRaises(OverflowError):
            Vector(sys.maxsize)

    def test_index_out_of_bounds_raises_index_error(self) -> None:
        vector = Vector([1.0, 2.0])

        with self.assertRaises(StrataxIndexError):
            _ = vector[2]

    def test_negative_index_raises_index_error(self) -> None:
        vector = Vector([1.0, 2.0])

        with self.assertRaises(StrataxIndexError):
            _ = vector[-1]


if __name__ == "__main__":
    unittest.main()
