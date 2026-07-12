from __future__ import annotations

import sys
import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from stratax import TypeError as StrataxTypeError, IndexError as StrataxIndexError
from stratax import Shape, ShapeError


class TestShapeInterfaceTests:
    def test_default_shape_is_empty(self) -> None:
        shape = Shape()

        assert shape.rank == 0
        assert shape.elements == 0
        assert shape.empty
        assert len(shape) == 0
        assert list(shape) == []
        assert repr(shape) == "()"

    def test_iterable_constructor_exposes_dimensions(self) -> None:
        shape = Shape([2, 3, 4])

        assert shape.rank == 3
        assert shape.elements == 24
        assert not shape.empty
        assert len(shape) == 3
        assert shape[0] == 2
        assert shape[1] == 3
        assert shape[2] == 4
        assert list(shape) == [2, 3, 4]
        assert repr(shape) == "(2, 3, 4)"

    def test_varargs_constructor_builds_shape(self) -> None:
        shape = Shape(2, 3, 4)

        assert shape.rank == 3
        assert shape.elements == 24
        assert list(shape) == [2, 3, 4]

    def test_generator_constructor_builds_shape(self) -> None:
        shape = Shape(dim for dim in (3, 2, 1))

        assert shape.rank == 3
        assert shape.elements == 6
        assert list(shape) == [3, 2, 1]

    def test_single_dimension_shape_repr_has_trailing_comma(self) -> None:
        shape = Shape([5])

        assert shape.rank == 1
        assert shape.elements == 5
        assert list(shape) == [5]
        assert repr(shape) == "(5,)"

    def test_zero_dimension_makes_zero_elements_but_keeps_rank(self) -> None:
        shape = Shape([2, 0, 4])

        assert shape.rank == 3
        assert shape.elements == 0
        assert not shape.empty
        assert list(shape) == [2, 0, 4]

    def test_copy_constructor_preserves_value(self) -> None:
        original = Shape([2, 3])
        copied = Shape(original)

        assert copied == original
        assert copied is not original
        assert list(copied) == [2, 3]

    def test_equality_and_inequality(self) -> None:
        assert Shape([2, 3]) == Shape(2, 3)
        assert Shape([2, 3]) != Shape([3, 2])
        assert Shape([2, 3]) != (2, 3)
        assert not Shape([2, 3]) == (2, 3)
        assert Shape([2, 3]) != (2, 3)

    def test_single_integer_is_not_a_valid_constructor_argument(self) -> None:
        with pytest.raises(StrataxTypeError):
            Shape(3)

    def test_non_iterable_constructor_argument_raises_type_error(self) -> None:
        with pytest.raises(StrataxTypeError):
            Shape(object())

    def test_negative_dimensions_raise_shape_error(self) -> None:
        with pytest.raises(ShapeError):
            Shape([-1])

    @pytest.mark.parametrize("value", [True, 1.5, "3"])
    def test_non_integer_dimensions_raise_type_error(self, value: object) -> None:
        with pytest.raises(StrataxTypeError):
            Shape([value])

    def test_single_zero_dimension_is_allowed(self) -> None:
        shape = Shape([0])

        assert shape.rank == 1
        assert shape.elements == 0
        assert list(shape) == [0]

    def test_shape_element_count_overflow_raises_overflow_error(self) -> None:
        with pytest.raises(OverflowError):
            Shape([sys.maxsize, 3])

    def test_dimension_cast_overflow_raises_overflow_error(self) -> None:
        with pytest.raises(OverflowError):
            Shape([sys.maxsize + 1])

    def test_dimension_index_out_of_bounds_raises_index_error(self) -> None:
        shape = Shape([2, 3])

        with pytest.raises(StrataxIndexError):
            _ = shape[2]

    def test_negative_dimension_index_raises_index_error(self) -> None:
        shape = Shape([2, 3])

        with pytest.raises(StrataxIndexError):
            _ = shape[-1]

