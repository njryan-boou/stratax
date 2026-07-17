from __future__ import annotations

import sys
import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from stratax import TypeError as StrataxTypeError, DimensionError, IndexError as StrataxIndexError
from stratax import Shape, Tensor, Vector, ZeroDivisionError as StrataxZeroDivisionError


class TestVectorInterfaceTests:
    def test_default_vector_is_empty_rank_one_vector(self) -> None:
        vector = Vector()

        assert vector.size == 0
        assert vector.rank == 1
        assert vector.empty
        assert len(vector) == 0
        assert list(vector.shape) == [0]
        assert vector.strides == [1]
        assert vector.tolist() == []
        assert list(vector) == []

    def test_size_constructor_builds_default_values(self) -> None:
        vector = Vector(3)

        assert vector.size == 3
        assert vector.rank == 1
        assert not vector.empty
        assert vector.shape == Shape([3])
        assert vector.strides == [1]
        assert len(vector) == 3
        assert vector.tolist() == [0.0, 0.0, 0.0]

    def test_size_value_constructor_fills_values(self) -> None:
        vector = Vector(4, 2.5)

        assert vector.size == 4
        assert vector.tolist() == [2.5, 2.5, 2.5, 2.5]

    def test_shape_constructor_builds_vector(self) -> None:
        vector = Vector(Shape([3]))

        assert vector.size == 3
        assert vector.shape == Shape([3])
        assert vector.tolist() == [0.0, 0.0, 0.0]

    def test_iterable_constructor_preserves_values(self) -> None:
        vector = Vector([1.0, 2.5, 3.0])

        assert vector.size == 3
        assert vector.shape == Shape([3])
        assert vector.tolist() == [1.0, 2.5, 3.0]
        assert list(vector) == [1.0, 2.5, 3.0]

    def test_generator_constructor_preserves_values(self) -> None:
        vector = Vector(float(value) for value in (1, 2, 3))

        assert vector.tolist() == [1.0, 2.0, 3.0]

    def test_copy_constructor_preserves_values(self) -> None:
        original = Vector([1.0, 2.0, 3.0])
        copied = Vector(original)

        assert copied is not original
        assert copied.tolist() == original.tolist()
        copied[0] = 9.0
        assert original.tolist() == [1.0, 2.0, 3.0]
        assert copied.tolist() == [9.0, 2.0, 3.0]

    def test_indexing_reads_and_writes_values(self) -> None:
        vector = Vector([1.0, 2.0, 3.0])

        assert vector[1] == 2.0
        vector[1] = 8.0
        assert vector[1] == 8.0
        assert vector.tolist() == [1.0, 8.0, 3.0]

    def test_slice_indexing_returns_vector(self) -> None:
        vector = Vector([1.0, 2.0, 3.0, 4.0])

        sliced = vector[1:3]

        assert isinstance(sliced, Vector)
        assert sliced.shape == Shape([2])
        assert sliced.tolist() == [2.0, 3.0]

    def test_slice_indexing_supports_step(self) -> None:
        vector = Vector([1.0, 2.0, 3.0, 4.0, 5.0])

        sliced = vector[::2]

        assert isinstance(sliced, Vector)
        assert sliced.shape == Shape([3])
        assert sliced.tolist() == [1.0, 3.0, 5.0]

    def test_slice_indexing_supports_negative_step(self) -> None:
        vector = Vector([1.0, 2.0, 3.0, 4.0, 5.0])

        sliced = vector[::-2]

        assert isinstance(sliced, Vector)
        assert sliced.shape == Shape([3])
        assert sliced.tolist() == [5.0, 3.0, 1.0]

    def test_fill_updates_all_values(self) -> None:
        vector = Vector([1.0, 2.0, 3.0])

        vector.fill(4.5)

        assert vector.tolist() == [4.5, 4.5, 4.5]

    def test_reshape_and_flatten(self) -> None:
        vector = Vector([1.0, 2.0, 3.0, 4.0])

        reshaped = vector.reshape(Shape(2, 2))
        reshaped_dims = vector.reshape([1, 4])
        flattened = vector.flatten()

        assert isinstance(reshaped, Tensor)
        assert reshaped.shape == Shape(2, 2)
        assert reshaped.tolist() == [1.0, 2.0, 3.0, 4.0]

        assert isinstance(reshaped_dims, Tensor)
        assert reshaped_dims.shape == Shape(1, 4)
        assert reshaped_dims.tolist() == [1.0, 2.0, 3.0, 4.0]

        assert isinstance(flattened, Vector)
        assert flattened.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_repr_returns_vector_text(self) -> None:
        assert repr(Vector([1.0, 2.0, 3.0])) == "[1, 2, 3]"

    def test_equality_and_inequality(self) -> None:
        assert Vector([1.0, 2.0, 3.0]) == Vector([1.0, 2.0, 3.0])
        assert Vector([1.0, 2.0, 3.0]) != Vector([1.0, 9.0, 3.0])
        assert Vector([1.0, 2.0, 3.0]) != Vector([1.0, 2.0])
        assert Vector([1.0, 2.0, 3.0]) != [1.0, 2.0, 3.0]

    def test_array_arithmetic(self) -> None:
        lhs = Vector([8.0, 12.0, 20.0])
        rhs = Vector([2.0, 3.0, 5.0])

        assert (lhs + rhs).tolist() == [10.0, 15.0, 25.0]
        assert (lhs - rhs).tolist() == [6.0, 9.0, 15.0]
        assert (lhs * rhs).tolist() == [16.0, 36.0, 100.0]
        assert (lhs / rhs).tolist() == [4.0, 4.0, 4.0]

        with pytest.raises(StrataxZeroDivisionError):
            _ = lhs / Vector([1.0, 0.0, 1.0])

    def test_scalar_arithmetic(self) -> None:
        vector = Vector([2.0, 4.0, 8.0])

        assert (vector + 2).tolist() == [4.0, 6.0, 10.0]
        assert (vector - 2).tolist() == [0.0, 2.0, 6.0]
        assert (vector * 2).tolist() == [4.0, 8.0, 16.0]
        assert (vector / 2).tolist() == [1.0, 2.0, 4.0]

        with pytest.raises(StrataxZeroDivisionError):
            _ = vector / 0

    def test_reverse_scalar_arithmetic(self) -> None:
        vector = Vector([2.0, 4.0, 8.0])

        assert (2 + vector).tolist() == [4.0, 6.0, 10.0]
        assert (20 - vector).tolist() == [18.0, 16.0, 12.0]
        assert (2 * vector).tolist() == [4.0, 8.0, 16.0]
        assert (32 / vector).tolist() == [16.0, 8.0, 4.0]

    def test_in_place_arithmetic(self) -> None:
        vector = Vector([2.0, 4.0, 8.0])

        vector += 2
        assert vector.tolist() == [4.0, 6.0, 10.0]

        vector -= Vector([1.0, 2.0, 3.0])
        assert vector.tolist() == [3.0, 4.0, 7.0]

        vector *= 2
        assert vector.tolist() == [6.0, 8.0, 14.0]

        vector /= Vector([3.0, 4.0, 7.0])
        assert vector.tolist() == [2.0, 2.0, 2.0]

    def test_unary_arithmetic(self) -> None:
        vector = Vector([1.0, -2.0, 3.0])

        assert (+vector).tolist() == [1.0, -2.0, 3.0]
        assert (-vector).tolist() == [-1.0, 2.0, -3.0]

    def test_invalid_constructor_argument_raises_type_error(self) -> None:
        with pytest.raises(StrataxTypeError):
            Vector(object())

    def test_too_many_constructor_arguments_raise_type_error(self) -> None:
        with pytest.raises(TypeError):
            Vector(1, 2, 3)

    def test_size_value_constructor_requires_integer_size(self) -> None:
        with pytest.raises(StrataxTypeError):
            Vector(1.5, 2.0)

        with pytest.raises(StrataxTypeError):
            Vector(True)

        with pytest.raises(StrataxTypeError):
            Vector(2, object())

    def test_iterable_constructor_requires_numeric_values(self) -> None:
        with pytest.raises(StrataxTypeError):
            Vector([1.0, object()])

        with pytest.raises(StrataxTypeError):
            Vector([True])

    def test_unsupported_arithmetic_operand_raises_type_error(self) -> None:
        vector = Vector([1.0, 2.0])

        with pytest.raises(TypeError):
            _ = vector + object()

        with pytest.raises(TypeError):
            _ = object() + vector

    def test_negative_size_raises_error(self) -> None:
        with pytest.raises(DimensionError):
            Vector(-1)

    def test_size_overflow_raises_overflow_error(self) -> None:
        with pytest.raises(OverflowError):
            Vector(sys.maxsize)

        with pytest.raises(OverflowError):
            Vector(sys.maxsize + 1)

    def test_index_out_of_bounds_raises_index_error(self) -> None:
        vector = Vector([1.0, 2.0])

        with pytest.raises(StrataxIndexError):
            _ = vector[2]

    def test_index_overflow_raises_overflow_error(self) -> None:
        vector = Vector([1.0, 2.0])

        with pytest.raises(OverflowError):
            _ = vector[sys.maxsize + 1]

    def test_negative_index_reads_and_writes_values(self) -> None:
        vector = Vector([1.0, 2.0])

        assert vector[-1] == 2.0
        vector[-1] = 9.0
        assert vector.tolist() == [1.0, 9.0]

