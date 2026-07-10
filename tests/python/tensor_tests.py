from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from stratax import DTypeError, IndexError as StrataxIndexError
from stratax import Shape, ShapeError, Tensor, ZeroDivisionError as StrataxZeroDivisionError


class TensorInterfaceTests(unittest.TestCase):
    def test_default_tensor_is_empty_rank_zero_tensor(self) -> None:
        tensor = Tensor()

        self.assertEqual(tensor.size, 0)
        self.assertEqual(tensor.rank, 0)
        self.assertTrue(tensor.empty)
        self.assertEqual(len(tensor), 0)
        self.assertEqual(tensor.shape, Shape())
        self.assertEqual(tensor.strides, [])
        self.assertEqual(tensor.tolist(), [])
        self.assertEqual(list(tensor), [])

    def test_shape_constructor_builds_default_values(self) -> None:
        tensor = Tensor(Shape(2, 3))

        self.assertEqual(tensor.size, 6)
        self.assertEqual(tensor.rank, 2)
        self.assertFalse(tensor.empty)
        self.assertEqual(tensor.shape, Shape(2, 3))
        self.assertEqual(tensor.strides, [3, 1])
        self.assertEqual(tensor.tolist(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_shape_value_constructor_fills_values(self) -> None:
        tensor = Tensor(Shape(2, 2), 3.5)

        self.assertEqual(tensor.size, 4)
        self.assertEqual(tensor.tolist(), [3.5, 3.5, 3.5, 3.5])

    def test_iterable_shape_constructor_builds_tensor(self) -> None:
        tensor = Tensor([2, 2, 2])

        self.assertEqual(tensor.size, 8)
        self.assertEqual(tensor.rank, 3)
        self.assertEqual(tensor.shape, Shape(2, 2, 2))
        self.assertEqual(tensor.strides, [4, 2, 1])

    def test_iterable_shape_value_constructor_fills_values(self) -> None:
        tensor = Tensor([2, 3], 6.0)

        self.assertEqual(tensor.shape, Shape(2, 3))
        self.assertEqual(tensor.tolist(), [6.0, 6.0, 6.0, 6.0, 6.0, 6.0])

    def test_generator_shape_constructor_builds_tensor(self) -> None:
        tensor = Tensor(dim for dim in (2, 1, 3))

        self.assertEqual(tensor.size, 6)
        self.assertEqual(tensor.shape, Shape(2, 1, 3))
        self.assertEqual(tensor.strides, [3, 3, 1])

    def test_copy_constructor_preserves_values(self) -> None:
        original = Tensor([2, 2], 1.5)
        copied = Tensor(original)

        self.assertIsNot(copied, original)
        self.assertEqual(copied.tolist(), original.tolist())
        copied[0] = 9.0
        self.assertEqual(original.tolist(), [1.5, 1.5, 1.5, 1.5])
        self.assertEqual(copied.tolist(), [9.0, 1.5, 1.5, 1.5])

    def test_flat_indexing_reads_and_writes_values(self) -> None:
        tensor = Tensor([2, 2], 1.0)

        self.assertEqual(tensor[2], 1.0)
        tensor[2] = 8.0
        self.assertEqual(tensor[2], 8.0)
        self.assertEqual(tensor.tolist(), [1.0, 1.0, 8.0, 1.0])

    def test_tuple_indexing_reads_and_writes_values(self) -> None:
        tensor = Tensor([2, 3], 0.0)

        tensor[1, 2] = 7.0

        self.assertEqual(tensor[1, 2], 7.0)
        self.assertEqual(tensor.tolist(), [0.0, 0.0, 0.0, 0.0, 0.0, 7.0])

    def test_fill_updates_all_values(self) -> None:
        tensor = Tensor([2, 2], 1.0)

        tensor.fill(4.0)

        self.assertEqual(tensor.tolist(), [4.0, 4.0, 4.0, 4.0])

    def test_repr_returns_tensor_text(self) -> None:
        tensor = Tensor([2, 2])
        tensor[0] = 1.0
        tensor[1] = 2.0
        tensor[2] = 3.0
        tensor[3] = 4.0

        self.assertEqual(repr(tensor), "[\n    [1, 2],\n    [3, 4]\n]")

    def test_equality_and_inequality(self) -> None:
        self.assertEqual(Tensor([2, 2], 1.0), Tensor([2, 2], 1.0))
        self.assertNotEqual(Tensor([2, 2], 1.0), Tensor([2, 2], 2.0))
        self.assertNotEqual(Tensor([2, 2], 1.0), Tensor([4], 1.0))
        self.assertNotEqual(Tensor([2, 2], 1.0), [1.0, 1.0, 1.0, 1.0])

    def test_array_arithmetic(self) -> None:
        lhs = Tensor([2, 2])
        rhs = Tensor([2, 2])

        for index, value in enumerate([8.0, 12.0, 20.0, 30.0]):
            lhs[index] = value

        for index, value in enumerate([2.0, 3.0, 5.0, 6.0]):
            rhs[index] = value

        self.assertEqual((lhs + rhs).tolist(), [10.0, 15.0, 25.0, 36.0])
        self.assertEqual((lhs - rhs).tolist(), [6.0, 9.0, 15.0, 24.0])
        self.assertEqual((lhs * rhs).tolist(), [16.0, 36.0, 100.0, 180.0])
        self.assertEqual((lhs / rhs).tolist(), [4.0, 4.0, 4.0, 5.0])

        with self.assertRaises(StrataxZeroDivisionError):
            _ = lhs / Tensor([2, 2], 0.0)

    def test_scalar_reverse_in_place_and_unary_arithmetic(self) -> None:
        tensor = Tensor([2, 2], 2.0)

        self.assertEqual((tensor + 3).tolist(), [5.0, 5.0, 5.0, 5.0])
        self.assertEqual((tensor - 1).tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual((tensor * 4).tolist(), [8.0, 8.0, 8.0, 8.0])
        self.assertEqual((tensor / 2).tolist(), [1.0, 1.0, 1.0, 1.0])
        with self.assertRaises(StrataxZeroDivisionError):
            _ = tensor / 0

        self.assertEqual((3 + tensor).tolist(), [5.0, 5.0, 5.0, 5.0])
        self.assertEqual((10 - tensor).tolist(), [8.0, 8.0, 8.0, 8.0])
        self.assertEqual((4 * tensor).tolist(), [8.0, 8.0, 8.0, 8.0])
        self.assertEqual((8 / tensor).tolist(), [4.0, 4.0, 4.0, 4.0])
        self.assertEqual((+tensor).tolist(), [2.0, 2.0, 2.0, 2.0])
        self.assertEqual((-tensor).tolist(), [-2.0, -2.0, -2.0, -2.0])

        tensor += 1
        self.assertEqual(tensor.tolist(), [3.0, 3.0, 3.0, 3.0])

        tensor -= Tensor([2, 2], 1.0)
        self.assertEqual(tensor.tolist(), [2.0, 2.0, 2.0, 2.0])

        tensor *= 4
        self.assertEqual(tensor.tolist(), [8.0, 8.0, 8.0, 8.0])

        tensor /= Tensor([2, 2], 2.0)
        self.assertEqual(tensor.tolist(), [4.0, 4.0, 4.0, 4.0])

    def test_invalid_constructor_argument_raises_type_error(self) -> None:
        with self.assertRaises(DTypeError):
            Tensor(object())

    def test_constructor_rejects_bad_arity_and_bad_shape_value_pair(self) -> None:
        with self.assertRaises(TypeError):
            Tensor([2], 1.0, 3.0)

        with self.assertRaises(DTypeError):
            Tensor(object(), 1.0)

    def test_unsupported_arithmetic_operand_raises_type_error(self) -> None:
        tensor = Tensor([2], 1.0)

        with self.assertRaises(TypeError):
            _ = tensor + object()

        with self.assertRaises(TypeError):
            _ = object() + tensor

    def test_negative_dimension_raises_error(self) -> None:
        with self.assertRaises(ShapeError):
            Tensor([-1])

    def test_size_overflow_raises_overflow_error(self) -> None:
        with self.assertRaises(OverflowError):
            Tensor([sys.maxsize, 2])

    def test_index_out_of_bounds_raises_index_error(self) -> None:
        tensor = Tensor([2, 2])

        with self.assertRaises(StrataxIndexError):
            _ = tensor[4]

    def test_negative_index_raises_index_error(self) -> None:
        tensor = Tensor([2, 2])

        with self.assertRaises(StrataxIndexError):
            _ = tensor[-1]

    def test_bad_tuple_index_rank_raises_index_error(self) -> None:
        tensor = Tensor([2, 2])

        with self.assertRaises(StrataxIndexError):
            _ = tensor[0, 0, 0]


if __name__ == "__main__":
    unittest.main()
