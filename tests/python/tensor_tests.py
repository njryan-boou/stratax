from __future__ import annotations

import sys
import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from stratax import TypeError as StrataxTypeError, IndexError as StrataxIndexError
from stratax import Shape, ShapeError, Tensor, ZeroDivisionError as StrataxZeroDivisionError
from stratax import Vector


class TestTensorInterfaceTests:
    def test_default_tensor_is_empty_rank_zero_tensor(self) -> None:
        tensor = Tensor()

        assert tensor.size == 0
        assert tensor.rank == 0
        assert tensor.empty
        assert len(tensor) == 0
        assert tensor.shape == Shape()
        assert tensor.strides == []
        assert tensor.tolist() == []
        assert list(tensor) == []

    def test_shape_constructor_builds_default_values(self) -> None:
        tensor = Tensor(Shape(2, 3))

        assert tensor.size == 6
        assert tensor.rank == 2
        assert not tensor.empty
        assert tensor.shape == Shape(2, 3)
        assert tensor.strides == [3, 1]
        assert tensor.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_shape_value_constructor_fills_values(self) -> None:
        tensor = Tensor(Shape(2, 2), 3.5)

        assert tensor.size == 4
        assert tensor.tolist() == [3.5, 3.5, 3.5, 3.5]

    def test_iterable_shape_constructor_builds_tensor(self) -> None:
        tensor = Tensor([2, 2, 2])

        assert tensor.size == 8
        assert tensor.rank == 3
        assert tensor.shape == Shape(2, 2, 2)
        assert tensor.strides == [4, 2, 1]

    def test_iterable_shape_value_constructor_fills_values(self) -> None:
        tensor = Tensor([2, 3], 6.0)

        assert tensor.shape == Shape(2, 3)
        assert tensor.tolist() == [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]

    def test_generator_shape_constructor_builds_tensor(self) -> None:
        tensor = Tensor(dim for dim in (2, 1, 3))

        assert tensor.size == 6
        assert tensor.shape == Shape(2, 1, 3)
        assert tensor.strides == [3, 3, 1]

    def test_copy_constructor_preserves_values(self) -> None:
        original = Tensor([2, 2], 1.5)
        copied = Tensor(original)

        assert copied is not original
        assert copied.tolist() == original.tolist()
        copied[0] = 9.0
        assert original.tolist() == [1.5, 1.5, 1.5, 1.5]
        assert copied.tolist() == [9.0, 1.5, 1.5, 1.5]

    def test_flat_indexing_reads_and_writes_values(self) -> None:
        tensor = Tensor([2, 2], 1.0)

        assert tensor[2] == 1.0
        tensor[2] = 8.0
        assert tensor[2] == 8.0
        assert tensor.tolist() == [1.0, 1.0, 8.0, 1.0]

    def test_tuple_indexing_reads_and_writes_values(self) -> None:
        tensor = Tensor([2, 3], 0.0)

        tensor[1, 2] = 7.0

        assert tensor[1, 2] == 7.0
        assert tensor.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 7.0]

    def test_slice_indexing_returns_tensor(self) -> None:
        tensor = Tensor([5])
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            tensor[index] = value

        sliced = tensor[1:4]

        assert isinstance(sliced, Tensor)
        assert sliced.shape == Shape([3])
        assert sliced.tolist() == [2.0, 3.0, 4.0]

    def test_mixed_tuple_slice_indexing_returns_tensor(self) -> None:
        tensor = Tensor([2, 3])
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            tensor[index] = value

        sliced = tensor[:, 1:3]

        assert isinstance(sliced, Tensor)
        assert sliced.shape == Shape([2, 2])
        assert sliced.tolist() == [2.0, 3.0, 5.0, 6.0]

    def test_integer_and_slice_tuple_indexing_returns_tensor(self) -> None:
        tensor = Tensor([2, 3])
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            tensor[index] = value

        sliced = tensor[1, 1:3]

        assert isinstance(sliced, Tensor)
        assert sliced.shape == Shape([1, 2])
        assert sliced.tolist() == [5.0, 6.0]

    def test_rank_mismatch_top_level_slice_raises_index_error(self) -> None:
        tensor = Tensor([2, 2], 1.0)

        with pytest.raises(StrataxIndexError):
            _ = tensor[:]

    def test_slice_indexing_supports_step(self) -> None:
        tensor = Tensor([5])
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            tensor[index] = value

        sliced = tensor[::2]

        assert isinstance(sliced, Tensor)
        assert sliced.shape == Shape([3])
        assert sliced.tolist() == [1.0, 3.0, 5.0]

    def test_slice_indexing_supports_negative_step(self) -> None:
        tensor = Tensor([5])
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            tensor[index] = value

        sliced = tensor[::-2]

        assert isinstance(sliced, Tensor)
        assert sliced.shape == Shape([3])
        assert sliced.tolist() == [5.0, 3.0, 1.0]

    def test_tuple_slice_indexing_supports_negative_step(self) -> None:
        tensor = Tensor([2, 3])
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            tensor[index] = value

        sliced = tensor[::-1, ::-2]

        assert isinstance(sliced, Tensor)
        assert sliced.shape == Shape([2, 2])
        assert sliced.tolist() == [6.0, 4.0, 3.0, 1.0]

    def test_fill_updates_all_values(self) -> None:
        tensor = Tensor([2, 2], 1.0)

        tensor.fill(4.0)

        assert tensor.tolist() == [4.0, 4.0, 4.0, 4.0]

    def test_reshape_accepts_shape_and_iterable(self) -> None:
        tensor = Tensor([2, 3])
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            tensor[index] = value

        reshaped_with_shape = tensor.reshape(Shape(3, 2))
        reshaped_with_iterable = tensor.reshape([1, 6])

        assert isinstance(reshaped_with_shape, Tensor)
        assert reshaped_with_shape.shape == Shape(3, 2)
        assert reshaped_with_shape.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        assert isinstance(reshaped_with_iterable, Tensor)
        assert reshaped_with_iterable.shape == Shape(1, 6)
        assert reshaped_with_iterable.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_flatten_returns_vector(self) -> None:
        tensor = Tensor([2, 2])
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0]):
            tensor[index] = value

        flattened = tensor.flatten()

        assert isinstance(flattened, Vector)
        assert flattened.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_repr_returns_tensor_text(self) -> None:
        tensor = Tensor([2, 2])
        tensor[0] = 1.0
        tensor[1] = 2.0
        tensor[2] = 3.0
        tensor[3] = 4.0

        assert repr(tensor) == "[\n    [1, 2],\n    [3, 4]\n]"

    def test_equality_and_inequality(self) -> None:
        assert Tensor([2, 2], 1.0) == Tensor([2, 2], 1.0)
        assert Tensor([2, 2], 1.0) != Tensor([2, 2], 2.0)
        assert Tensor([2, 2], 1.0) != Tensor([4], 1.0)
        assert Tensor([2, 2], 1.0) != [1.0, 1.0, 1.0, 1.0]

    def test_array_arithmetic(self) -> None:
        lhs = Tensor([2, 2])
        rhs = Tensor([2, 2])

        for index, value in enumerate([8.0, 12.0, 20.0, 30.0]):
            lhs[index] = value

        for index, value in enumerate([2.0, 3.0, 5.0, 6.0]):
            rhs[index] = value

        assert (lhs + rhs).tolist() == [10.0, 15.0, 25.0, 36.0]
        assert (lhs - rhs).tolist() == [6.0, 9.0, 15.0, 24.0]
        assert (lhs * rhs).tolist() == [16.0, 36.0, 100.0, 180.0]
        assert (lhs / rhs).tolist() == [4.0, 4.0, 4.0, 5.0]

        with pytest.raises(StrataxZeroDivisionError):
            _ = lhs / Tensor([2, 2], 0.0)

    def test_scalar_reverse_in_place_and_unary_arithmetic(self) -> None:
        tensor = Tensor([2, 2], 2.0)

        assert (tensor + 3).tolist() == [5.0, 5.0, 5.0, 5.0]
        assert (tensor - 1).tolist() == [1.0, 1.0, 1.0, 1.0]
        assert (tensor * 4).tolist() == [8.0, 8.0, 8.0, 8.0]
        assert (tensor / 2).tolist() == [1.0, 1.0, 1.0, 1.0]
        with pytest.raises(StrataxZeroDivisionError):
            _ = tensor / 0

        assert (3 + tensor).tolist() == [5.0, 5.0, 5.0, 5.0]
        assert (10 - tensor).tolist() == [8.0, 8.0, 8.0, 8.0]
        assert (4 * tensor).tolist() == [8.0, 8.0, 8.0, 8.0]
        assert (8 / tensor).tolist() == [4.0, 4.0, 4.0, 4.0]
        assert (+tensor).tolist() == [2.0, 2.0, 2.0, 2.0]
        assert (-tensor).tolist() == [-2.0, -2.0, -2.0, -2.0]

        tensor += 1
        assert tensor.tolist() == [3.0, 3.0, 3.0, 3.0]

        tensor -= Tensor([2, 2], 1.0)
        assert tensor.tolist() == [2.0, 2.0, 2.0, 2.0]

        tensor *= 4
        assert tensor.tolist() == [8.0, 8.0, 8.0, 8.0]

        tensor /= Tensor([2, 2], 2.0)
        assert tensor.tolist() == [4.0, 4.0, 4.0, 4.0]

    def test_invalid_constructor_argument_raises_type_error(self) -> None:
        with pytest.raises(StrataxTypeError):
            Tensor(object())

    def test_constructor_rejects_bad_arity_and_bad_shape_value_pair(self) -> None:
        with pytest.raises(TypeError):
            Tensor([2], 1.0, 3.0)

        with pytest.raises(StrataxTypeError):
            Tensor(object(), 1.0)

        with pytest.raises(StrataxTypeError):
            Tensor([2], object())

        with pytest.raises(StrataxTypeError):
            Tensor([True])

        with pytest.raises(StrataxTypeError):
            Tensor("23")

    def test_unsupported_arithmetic_operand_raises_type_error(self) -> None:
        tensor = Tensor([2], 1.0)

        with pytest.raises(TypeError):
            _ = tensor + object()

        with pytest.raises(TypeError):
            _ = object() + tensor

    def test_negative_dimension_raises_error(self) -> None:
        with pytest.raises(ShapeError):
            Tensor([-1])

    def test_size_overflow_raises_overflow_error(self) -> None:
        with pytest.raises(OverflowError):
            Tensor([sys.maxsize, 2])

        with pytest.raises(OverflowError):
            Tensor([sys.maxsize + 1])

    def test_index_out_of_bounds_raises_index_error(self) -> None:
        tensor = Tensor([2, 2])

        with pytest.raises(StrataxIndexError):
            _ = tensor[4]

    def test_negative_index_reads_and_writes_values(self) -> None:
        tensor = Tensor([2, 2])

        assert tensor[-1] == 0.0
        tensor[-1] = 7.0
        assert tensor.tolist() == [0.0, 0.0, 0.0, 7.0]

    def test_index_overflow_raises_overflow_error(self) -> None:
        tensor = Tensor([2, 2])

        with pytest.raises(OverflowError):
            _ = tensor[sys.maxsize + 1]

    def test_bad_tuple_index_rank_raises_index_error(self) -> None:
        tensor = Tensor([2, 2])

        with pytest.raises(StrataxIndexError):
            _ = tensor[0, 0, 0]

