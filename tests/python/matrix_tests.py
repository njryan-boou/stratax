from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from stratax import DTypeError, DimensionError, IndexError as StrataxIndexError
from stratax import Matrix, Shape, StrataxError, ZeroDivisionError as StrataxZeroDivisionError


class MatrixInterfaceTests(unittest.TestCase):
    def test_default_matrix_is_empty_rank_two_matrix(self) -> None:
        matrix = Matrix()

        self.assertEqual(matrix.size, 0)
        self.assertEqual(matrix.rank, 2)
        self.assertTrue(matrix.empty)
        self.assertEqual(matrix.rows, 0)
        self.assertEqual(matrix.cols, 0)
        self.assertEqual(len(matrix), 0)
        self.assertEqual(matrix.shape, Shape(0, 0))
        self.assertEqual(matrix.strides, [0, 1])
        self.assertEqual(matrix.tolist(), [])
        self.assertEqual(list(matrix), [])

    def test_rows_cols_constructor_builds_default_values(self) -> None:
        matrix = Matrix(2, 3)

        self.assertEqual(matrix.size, 6)
        self.assertEqual(matrix.rank, 2)
        self.assertFalse(matrix.empty)
        self.assertEqual(matrix.rows, 2)
        self.assertEqual(matrix.cols, 3)
        self.assertEqual(len(matrix), 2)
        self.assertEqual(matrix.shape, Shape(2, 3))
        self.assertEqual(matrix.strides, [3, 1])
        self.assertEqual(
            matrix.tolist(),
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )

    def test_rows_cols_value_constructor_fills_values(self) -> None:
        matrix = Matrix(2, 2, 7.5)

        self.assertEqual(matrix.tolist(), [[7.5, 7.5], [7.5, 7.5]])

    def test_shape_constructor_builds_matrix(self) -> None:
        matrix = Matrix(Shape(2, 2))

        self.assertEqual(matrix.size, 4)
        self.assertEqual(matrix.shape, Shape(2, 2))
        self.assertEqual(matrix.tolist(), [[0.0, 0.0], [0.0, 0.0]])

    def test_nested_iterable_constructor_preserves_values(self) -> None:
        matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])

        self.assertEqual(matrix.size, 4)
        self.assertEqual(matrix.rows, 2)
        self.assertEqual(matrix.cols, 2)
        self.assertEqual(matrix.shape, Shape(2, 2))
        self.assertEqual(matrix.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(list(matrix), [1.0, 2.0, 3.0, 4.0])

    def test_copy_constructor_preserves_values(self) -> None:
        original = Matrix([[1.0, 2.0], [3.0, 4.0]])
        copied = Matrix(original)

        self.assertIsNot(copied, original)
        self.assertEqual(copied.tolist(), original.tolist())
        copied[0, 0] = 9.0
        self.assertEqual(original.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(copied.tolist(), [[9.0, 2.0], [3.0, 4.0]])

    def test_tuple_indexing_reads_and_writes_values(self) -> None:
        matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])

        self.assertEqual(matrix[1, 0], 3.0)
        matrix[1, 0] = 8.0
        self.assertEqual(matrix[1, 0], 8.0)
        self.assertEqual(matrix.tolist(), [[1.0, 2.0], [8.0, 4.0]])

    def test_fill_updates_all_values(self) -> None:
        matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])

        matrix.fill(5.0)

        self.assertEqual(matrix.tolist(), [[5.0, 5.0], [5.0, 5.0]])

    def test_repr_returns_matrix_text(self) -> None:
        self.assertEqual(repr(Matrix([[1.0, 2.0], [3.0, 4.0]])), "[\n    [1, 2]\n    [3, 4]\n]")

    def test_equality_and_inequality(self) -> None:
        self.assertEqual(
            Matrix([[1.0, 2.0], [3.0, 4.0]]),
            Matrix([[1.0, 2.0], [3.0, 4.0]]),
        )
        self.assertNotEqual(
            Matrix([[1.0, 2.0], [3.0, 4.0]]),
            Matrix([[1.0, 2.0], [3.0, 9.0]]),
        )
        self.assertNotEqual(
            Matrix([[1.0, 2.0], [3.0, 4.0]]),
            Matrix([[1.0, 2.0, 3.0]]),
        )
        self.assertNotEqual(
            Matrix([[1.0, 2.0], [3.0, 4.0]]),
            [[1.0, 2.0], [3.0, 4.0]],
        )

    def test_array_arithmetic(self) -> None:
        lhs = Matrix([[8.0, 12.0], [20.0, 30.0]])
        rhs = Matrix([[2.0, 3.0], [5.0, 6.0]])

        self.assertEqual((lhs + rhs).tolist(), [[10.0, 15.0], [25.0, 36.0]])
        self.assertEqual((lhs - rhs).tolist(), [[6.0, 9.0], [15.0, 24.0]])
        self.assertEqual((lhs * rhs).tolist(), [[16.0, 36.0], [100.0, 180.0]])
        self.assertEqual((lhs / rhs).tolist(), [[4.0, 4.0], [4.0, 5.0]])

        with self.assertRaises(StrataxZeroDivisionError):
            _ = lhs / Matrix([[1.0, 0.0], [1.0, 1.0]])

    def test_scalar_and_reverse_scalar_arithmetic(self) -> None:
        matrix = Matrix([[2.0, 4.0], [8.0, 16.0]])

        self.assertEqual((matrix + 2).tolist(), [[4.0, 6.0], [10.0, 18.0]])
        self.assertEqual((matrix - 2).tolist(), [[0.0, 2.0], [6.0, 14.0]])
        self.assertEqual((matrix * 2).tolist(), [[4.0, 8.0], [16.0, 32.0]])
        self.assertEqual((matrix / 2).tolist(), [[1.0, 2.0], [4.0, 8.0]])
        with self.assertRaises(StrataxZeroDivisionError):
            _ = matrix / 0

        self.assertEqual((2 + matrix).tolist(), [[4.0, 6.0], [10.0, 18.0]])
        self.assertEqual((20 - matrix).tolist(), [[18.0, 16.0], [12.0, 4.0]])
        self.assertEqual((2 * matrix).tolist(), [[4.0, 8.0], [16.0, 32.0]])
        self.assertEqual((32 / matrix).tolist(), [[16.0, 8.0], [4.0, 2.0]])

    def test_in_place_and_unary_arithmetic(self) -> None:
        matrix = Matrix([[2.0, -4.0], [8.0, -16.0]])

        self.assertEqual((+matrix).tolist(), [[2.0, -4.0], [8.0, -16.0]])
        self.assertEqual((-matrix).tolist(), [[-2.0, 4.0], [-8.0, 16.0]])

        matrix += 2
        self.assertEqual(matrix.tolist(), [[4.0, -2.0], [10.0, -14.0]])

        matrix -= Matrix([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(matrix.tolist(), [[3.0, -4.0], [7.0, -18.0]])

        matrix *= Matrix([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(matrix.tolist(), [[3.0, -8.0], [21.0, -72.0]])

        matrix /= 3
        self.assertEqual(matrix.tolist(), [[1.0, -2.6666666666666665], [7.0, -24.0]])

    def test_invalid_constructor_argument_raises_type_error(self) -> None:
        with self.assertRaises(DTypeError):
            Matrix(object())

    def test_constructor_rejects_bad_arity_and_non_integer_dimensions(self) -> None:
        with self.assertRaises(TypeError):
            Matrix(1, 2, 3, 4)

        with self.assertRaises(DTypeError):
            Matrix(1.5, 2)

        with self.assertRaises(DTypeError):
            Matrix(1, 2.5)

    def test_unsupported_arithmetic_operand_raises_type_error(self) -> None:
        matrix = Matrix([[1.0, 2.0]])

        with self.assertRaises(TypeError):
            _ = matrix + object()

        with self.assertRaises(TypeError):
            _ = object() + matrix

    def test_jagged_rows_raise_error(self) -> None:
        with self.assertRaises(StrataxError):
            Matrix([[1.0], [2.0, 3.0]])

    def test_index_out_of_bounds_raises_index_error(self) -> None:
        matrix = Matrix([[1.0, 2.0]])

        with self.assertRaises(StrataxIndexError):
            _ = matrix[1, 0]

    def test_negative_index_raises_index_error(self) -> None:
        matrix = Matrix([[1.0, 2.0]])

        with self.assertRaises(StrataxIndexError):
            _ = matrix[-1, 0]

    def test_bad_index_rank_raises_index_error(self) -> None:
        matrix = Matrix([[1.0, 2.0]])

        with self.assertRaises(StrataxIndexError):
            _ = matrix[0, 0, 0]

    def test_negative_dimensions_raise_dimension_error(self) -> None:
        with self.assertRaises(DimensionError):
            Matrix(-1, 2)

    def test_size_overflow_raises_overflow_error(self) -> None:
        with self.assertRaises(OverflowError):
            Matrix(sys.maxsize, 2)


if __name__ == "__main__":
    unittest.main()
