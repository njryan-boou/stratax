from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import stratax
from stratax import Matrix, Tensor, Vector


class TestReductionsApi:
    def test_package_exports_all_reduction_functions(self) -> None:
        exported = set(stratax.__all__)

        assert {"sum", "prod", "max", "min", "argmax", "argmin", "mean", "var", "std"}.issubset(exported)

    def test_global_vector_reductions(self) -> None:
        vector = Vector([1.0, 2.0, 3.0])

        assert stratax.sum(vector) == 6.0
        assert stratax.prod(vector) == 6.0
        assert stratax.max(vector) == 3.0
        assert stratax.min(vector) == 1.0
        assert stratax.argmax(vector) == 2
        assert stratax.argmin(vector) == 0
        assert stratax.mean(vector) == 2.0
        assert stratax.var(vector) == 2.0 / 3.0
        assert stratax.std(vector) == (2.0 / 3.0) ** 0.5

    def test_global_matrix_reductions(self) -> None:
        matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])

        assert stratax.sum(matrix) == 10.0
        assert stratax.prod(matrix) == 24.0
        assert stratax.max(matrix) == 4.0
        assert stratax.min(matrix) == 1.0
        assert stratax.argmax(matrix) == 3
        assert stratax.argmin(matrix) == 0
        assert stratax.mean(matrix) == 2.5
        assert stratax.var(matrix) == 1.25
        assert stratax.std(matrix) == 1.25 ** 0.5

    def test_axis_reductions_support_keepdims(self) -> None:
        matrix = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        assert stratax.sum(matrix, 0, True).tolist() == [5.0, 7.0, 9.0]
        assert stratax.sum(matrix, 1, True).tolist() == [6.0, 15.0]
        assert stratax.mean(matrix, 0, True).tolist() == [2.5, 3.5, 4.5]
        assert stratax.mean(matrix, 1, True).tolist() == [2.0, 5.0]

    def test_negative_axis_matches_positive_axis(self) -> None:
        matrix = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        assert stratax.sum(matrix, -1).tolist() == stratax.sum(matrix, 1).tolist()
        assert stratax.mean(matrix, -1, True).tolist() == stratax.mean(matrix, 1, True).tolist()

    def test_axis_reductions_return_tensor_results(self) -> None:
        tensor = Tensor([2, 2, 2])

        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]):
            tensor[index] = value

        result = stratax.sum(tensor, 1, True)

        assert isinstance(result, Tensor)
        assert result.shape == tensor.shape.__class__([2, 1, 2])
        assert result.tolist() == [4.0, 6.0, 12.0, 14.0]

    def test_tensor_negative_axis_matches_last_axis(self) -> None:
        tensor = Tensor([2, 2, 2])
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]):
            tensor[index] = value

        assert stratax.sum(tensor, -1).tolist() == stratax.sum(tensor, 2).tolist()