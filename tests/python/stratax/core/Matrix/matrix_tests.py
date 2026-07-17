from __future__ import annotations

import sys
import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from stratax import TypeError as StrataxTypeError, DimensionError, IndexError as StrataxIndexError
from stratax import Matrix, Shape, StrataxError, Tensor, Vector, ZeroDivisionError as StrataxZeroDivisionError


class TestMatrixInterfaceTests:
    def test_default_matrix_is_empty_rank_two_matrix(self) -> None:
        matrix = Matrix()

        assert matrix.size == 0
        assert matrix.rank == 2
        assert matrix.empty
        assert matrix.rows == 0
        assert matrix.cols == 0
        assert len(matrix) == 0
        assert list(matrix.shape) == [0, 0]
        assert matrix.strides == [0, 1]
        assert matrix.tolist() == []
        assert list(matrix) == []

    def test_rows_cols_constructor_builds_default_values(self) -> None:
        matrix = Matrix(2, 3)

        assert matrix.size == 6
        assert matrix.rank == 2
        assert not matrix.empty
        assert matrix.rows == 2
        assert matrix.cols == 3
        assert len(matrix) == 2
        assert matrix.shape == Shape(2, 3)
        assert matrix.strides == [3, 1]
        assert matrix.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]

    def test_rows_cols_value_constructor_fills_values(self) -> None:
        matrix = Matrix(2, 2, 7.5)

        assert matrix.tolist() == [[7.5, 7.5], [7.5, 7.5]]

    def test_shape_constructor_builds_matrix(self) -> None:
        matrix = Matrix(Shape(2, 2))

        assert matrix.size == 4
        assert matrix.shape == Shape(2, 2)
        assert matrix.tolist() == [[0.0, 0.0], [0.0, 0.0]]

    def test_nested_iterable_constructor_preserves_values(self) -> None:
        matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])

        assert matrix.size == 4
        assert matrix.rows == 2
        assert matrix.cols == 2
        assert matrix.shape == Shape(2, 2)
        assert matrix.tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert list(matrix) == [1.0, 2.0, 3.0, 4.0]

    def test_copy_constructor_preserves_values(self) -> None:
        original = Matrix([[1.0, 2.0], [3.0, 4.0]])
        copied = Matrix(original)

        assert copied is not original
        assert copied.tolist() == original.tolist()
        copied[0, 0] = 9.0
        assert original.tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert copied.tolist() == [[9.0, 2.0], [3.0, 4.0]]

    def test_tuple_indexing_reads_and_writes_values(self) -> None:
        matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])

        assert matrix[1, 0] == 3.0
        matrix[1, 0] = 8.0
        assert matrix[1, 0] == 8.0
        assert matrix.tolist() == [[1.0, 2.0], [8.0, 4.0]]

    def test_slice_indexing_returns_matrix(self) -> None:
        matrix = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        sliced = matrix[:, 1:3]

        assert isinstance(sliced, Matrix)
        assert sliced.shape == Shape(2, 2)
        assert sliced.tolist() == [[2.0, 3.0], [5.0, 6.0]]

    def test_mixed_integer_and_slice_indexing_returns_matrix(self) -> None:
        matrix = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        sliced = matrix[1, :]

        assert isinstance(sliced, Matrix)
        assert sliced.shape == Shape(1, 3)
        assert sliced.tolist() == [[4.0, 5.0, 6.0]]

    def test_slice_indexing_supports_step(self) -> None:
        matrix = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        sliced = matrix[:, ::2]

        assert isinstance(sliced, Matrix)
        assert sliced.shape == Shape(2, 2)
        assert sliced.tolist() == [[1.0, 3.0], [4.0, 6.0]]

    def test_slice_indexing_supports_negative_step(self) -> None:
        matrix = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        sliced = matrix[::-1, ::-2]

        assert isinstance(sliced, Matrix)
        assert sliced.shape == Shape(2, 2)
        assert sliced.tolist() == [[6.0, 4.0], [3.0, 1.0]]

    def test_fill_updates_all_values(self) -> None:
        matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])

        matrix.fill(5.0)

        assert matrix.tolist() == [[5.0, 5.0], [5.0, 5.0]]

    def test_reshape_and_flatten(self) -> None:
        matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])

        reshaped = matrix.reshape(Shape(1, 4))
        reshaped_dims = matrix.reshape([4, 1])
        flattened = matrix.flatten()

        assert isinstance(reshaped, Tensor)
        assert reshaped.shape == Shape(1, 4)
        assert reshaped.tolist() == [1.0, 2.0, 3.0, 4.0]

        assert isinstance(reshaped_dims, Tensor)
        assert reshaped_dims.shape == Shape(4, 1)
        assert reshaped_dims.tolist() == [1.0, 2.0, 3.0, 4.0]

        assert isinstance(flattened, Vector)
        assert flattened.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_repr_returns_matrix_text(self) -> None:
        assert repr(Matrix([[1.0, 2.0], [3.0, 4.0]])) == "[\n    [1, 2]\n    [3, 4]\n]"

    def test_equality_and_inequality(self) -> None:
        assert Matrix([[1.0, 2.0], [3.0, 4.0]]) == Matrix([[1.0, 2.0], [3.0, 4.0]])
        assert Matrix([[1.0, 2.0], [3.0, 4.0]]) != Matrix([[1.0, 2.0], [3.0, 9.0]])
        assert Matrix([[1.0, 2.0], [3.0, 4.0]]) != Matrix([[1.0, 2.0, 3.0]])
        assert Matrix([[1.0, 2.0], [3.0, 4.0]]) != [[1.0, 2.0], [3.0, 4.0]]

    def test_array_arithmetic(self) -> None:
        lhs = Matrix([[8.0, 12.0], [20.0, 30.0]])
        rhs = Matrix([[2.0, 3.0], [5.0, 6.0]])

        assert (lhs + rhs).tolist() == [[10.0, 15.0], [25.0, 36.0]]
        assert (lhs - rhs).tolist() == [[6.0, 9.0], [15.0, 24.0]]
        assert (lhs * rhs).tolist() == [[16.0, 36.0], [100.0, 180.0]]
        assert (lhs / rhs).tolist() == [[4.0, 4.0], [4.0, 5.0]]

        with pytest.raises(StrataxZeroDivisionError):
            _ = lhs / Matrix([[1.0, 0.0], [1.0, 1.0]])

    def test_scalar_and_reverse_scalar_arithmetic(self) -> None:
        matrix = Matrix([[2.0, 4.0], [8.0, 16.0]])

        assert (matrix + 2).tolist() == [[4.0, 6.0], [10.0, 18.0]]
        assert (matrix - 2).tolist() == [[0.0, 2.0], [6.0, 14.0]]
        assert (matrix * 2).tolist() == [[4.0, 8.0], [16.0, 32.0]]
        assert (matrix / 2).tolist() == [[1.0, 2.0], [4.0, 8.0]]
        with pytest.raises(StrataxZeroDivisionError):
            _ = matrix / 0

        assert (2 + matrix).tolist() == [[4.0, 6.0], [10.0, 18.0]]
        assert (20 - matrix).tolist() == [[18.0, 16.0], [12.0, 4.0]]
        assert (2 * matrix).tolist() == [[4.0, 8.0], [16.0, 32.0]]
        assert (32 / matrix).tolist() == [[16.0, 8.0], [4.0, 2.0]]

    def test_in_place_and_unary_arithmetic(self) -> None:
        matrix = Matrix([[2.0, -4.0], [8.0, -16.0]])

        assert (+matrix).tolist() == [[2.0, -4.0], [8.0, -16.0]]
        assert (-matrix).tolist() == [[-2.0, 4.0], [-8.0, 16.0]]

        matrix += 2
        assert matrix.tolist() == [[4.0, -2.0], [10.0, -14.0]]

        matrix -= Matrix([[1.0, 2.0], [3.0, 4.0]])
        assert matrix.tolist() == [[3.0, -4.0], [7.0, -18.0]]

        matrix *= Matrix([[1.0, 2.0], [3.0, 4.0]])
        assert matrix.tolist() == [[3.0, -8.0], [21.0, -72.0]]

        matrix /= 3
        assert matrix.tolist() == [[1.0, -2.6666666666666665], [7.0, -24.0]]

    def test_invalid_constructor_argument_raises_type_error(self) -> None:
        with pytest.raises(StrataxTypeError):
            Matrix(object())

    def test_constructor_rejects_bad_arity_and_non_integer_dimensions(self) -> None:
        with pytest.raises(TypeError):
            Matrix(1, 2, 3, 4)

        with pytest.raises(StrataxTypeError):
            Matrix(1.5, 2)

        with pytest.raises(StrataxTypeError):
            Matrix(1, 2.5)

        with pytest.raises(StrataxTypeError):
            Matrix(True, 2)

        with pytest.raises(StrataxTypeError):
            Matrix(2, 2, object())

    def test_nested_iterable_constructor_requires_numeric_values(self) -> None:
        with pytest.raises(StrataxTypeError):
            Matrix([[1.0, object()]])

        with pytest.raises(StrataxTypeError):
            Matrix([[True]])

    def test_unsupported_arithmetic_operand_raises_type_error(self) -> None:
        matrix = Matrix([[1.0, 2.0]])

        with pytest.raises(TypeError):
            _ = matrix + object()

        with pytest.raises(TypeError):
            _ = object() + matrix

    def test_jagged_rows_raise_error(self) -> None:
        with pytest.raises(StrataxError):
            Matrix([[1.0], [2.0, 3.0]])

    def test_index_out_of_bounds_raises_index_error(self) -> None:
        matrix = Matrix([[1.0, 2.0]])

        with pytest.raises(StrataxIndexError):
            _ = matrix[1, 0]

    def test_negative_index_reads_and_writes_values(self) -> None:
        matrix = Matrix([[1.0, 2.0]])

        assert matrix[-1, -1] == 2.0
        matrix[-1, -1] = 8.0
        assert matrix.tolist() == [[1.0, 8.0]]

    def test_bad_index_rank_raises_index_error(self) -> None:
        matrix = Matrix([[1.0, 2.0]])

        with pytest.raises(StrataxIndexError):
            _ = matrix[0, 0, 0]

    def test_negative_dimensions_raise_dimension_error(self) -> None:
        with pytest.raises(DimensionError):
            Matrix(-1, 2)

    def test_size_overflow_raises_overflow_error(self) -> None:
        with pytest.raises(OverflowError):
            Matrix(sys.maxsize, 2)

        with pytest.raises(OverflowError):
            Matrix(sys.maxsize + 1, 1)

    def test_index_overflow_raises_overflow_error(self) -> None:
        matrix = Matrix([[1.0, 2.0]])

        with pytest.raises(OverflowError):
            _ = matrix[sys.maxsize + 1, 0]

