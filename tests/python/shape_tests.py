from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from stratax import DTypeError, IndexError as StrataxIndexError
from stratax import Shape, ShapeError


class ShapeInterfaceTests(unittest.TestCase):
    def test_default_shape_is_empty(self) -> None:
        shape = Shape()

        self.assertEqual(shape.rank, 0)
        self.assertEqual(shape.elements, 0)
        self.assertTrue(shape.empty)
        self.assertEqual(len(shape), 0)
        self.assertEqual(list(shape), [])
        self.assertEqual(repr(shape), "()")

    def test_iterable_constructor_exposes_dimensions(self) -> None:
        shape = Shape([2, 3, 4])

        self.assertEqual(shape.rank, 3)
        self.assertEqual(shape.elements, 24)
        self.assertFalse(shape.empty)
        self.assertEqual(len(shape), 3)
        self.assertEqual(shape[0], 2)
        self.assertEqual(shape[1], 3)
        self.assertEqual(shape[2], 4)
        self.assertEqual(list(shape), [2, 3, 4])
        self.assertEqual(repr(shape), "(2, 3, 4)")

    def test_varargs_constructor_builds_shape(self) -> None:
        shape = Shape(2, 3, 4)

        self.assertEqual(shape.rank, 3)
        self.assertEqual(shape.elements, 24)
        self.assertEqual(list(shape), [2, 3, 4])

    def test_generator_constructor_builds_shape(self) -> None:
        shape = Shape(dim for dim in (3, 2, 1))

        self.assertEqual(shape.rank, 3)
        self.assertEqual(shape.elements, 6)
        self.assertEqual(list(shape), [3, 2, 1])

    def test_single_dimension_shape_repr_has_trailing_comma(self) -> None:
        shape = Shape([5])

        self.assertEqual(shape.rank, 1)
        self.assertEqual(shape.elements, 5)
        self.assertEqual(list(shape), [5])
        self.assertEqual(repr(shape), "(5,)")

    def test_zero_dimension_makes_zero_elements_but_keeps_rank(self) -> None:
        shape = Shape([2, 0, 4])

        self.assertEqual(shape.rank, 3)
        self.assertEqual(shape.elements, 0)
        self.assertFalse(shape.empty)
        self.assertEqual(list(shape), [2, 0, 4])

    def test_copy_constructor_preserves_value(self) -> None:
        original = Shape([2, 3])
        copied = Shape(original)

        self.assertEqual(copied, original)
        self.assertIsNot(copied, original)
        self.assertEqual(list(copied), [2, 3])

    def test_equality_and_inequality(self) -> None:
        self.assertEqual(Shape([2, 3]), Shape(2, 3))
        self.assertNotEqual(Shape([2, 3]), Shape([3, 2]))
        self.assertNotEqual(Shape([2, 3]), (2, 3))
        self.assertFalse(Shape([2, 3]) == (2, 3))
        self.assertTrue(Shape([2, 3]) != (2, 3))

    def test_single_integer_is_not_a_valid_constructor_argument(self) -> None:
        with self.assertRaises(DTypeError):
            Shape(3)

    def test_non_iterable_constructor_argument_raises_type_error(self) -> None:
        with self.assertRaises(DTypeError):
            Shape(object())

    def test_negative_dimensions_raise_shape_error(self) -> None:
        with self.assertRaises(ShapeError):
            Shape([-1])

    def test_shape_element_count_overflow_raises_overflow_error(self) -> None:
        with self.assertRaises(OverflowError):
            Shape([sys.maxsize, 3])

    def test_dimension_index_out_of_bounds_raises_index_error(self) -> None:
        shape = Shape([2, 3])

        with self.assertRaises(StrataxIndexError):
            _ = shape[2]

    def test_negative_dimension_index_raises_index_error(self) -> None:
        shape = Shape([2, 3])

        with self.assertRaises(StrataxIndexError):
            _ = shape[-1]


if __name__ == "__main__":
    unittest.main()
