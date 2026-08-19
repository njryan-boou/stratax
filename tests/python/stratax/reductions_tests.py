from __future__ import annotations

import sys
from pathlib import Path


ROOT = next(candidate for candidate in Path(__file__).resolve().parents if (candidate / "python" / "stratax").exists())
sys.path.insert(0, str(ROOT / "python"))

import stratax as _core
from stratax import Matrix, Tensor, Vector


class TestReductionsApi:
    def test_package_exports_all_reduction_functions(self) -> None:
        exported = set(_core.__all__)

        assert {"sum", "prod", "max", "min", "argmax", "argmin", "mean", "var", "std"}.issubset(exported)

    def test_global_vector_reductions(self) -> None:
        vector = Vector([1.0, 2.0, 3.0])

        assert _core.sum(vector) == 6.0
        assert _core.prod(vector) == 6.0
        assert _core.max(vector) == 3.0
        assert _core.min(vector) == 1.0
        assert _core.argmax(vector) == 2
        assert _core.argmin(vector) == 0
        assert _core.mean(vector) == 2.0
        assert _core.var(vector) == 2.0 / 3.0
        assert _core.std(vector) == (2.0 / 3.0) ** 0.5

    def test_global_matrix_reductions(self) -> None:
        matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])

        assert _core.sum(matrix) == 10.0
        assert _core.prod(matrix) == 24.0
        assert _core.max(matrix) == 4.0
        assert _core.min(matrix) == 1.0
        assert _core.argmax(matrix) == 3
        assert _core.argmin(matrix) == 0
        assert _core.mean(matrix) == 2.5
        assert _core.var(matrix) == 1.25
        assert _core.std(matrix) == 1.25 ** 0.5

    def test_axis_reductions_support_keepdims(self) -> None:
        matrix = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        assert _core.sum(matrix, 0, True).tolist() == [[5.0, 7.0, 9.0]]
        assert _core.sum(matrix, 1, True).tolist() == [[6.0], [15.0]]
        assert _core.mean(matrix, 0, True).tolist() == [[2.5, 3.5, 4.5]]
        assert _core.mean(matrix, 1, True).tolist() == [[2.0], [5.0]]

    def test_negative_axis_matches_positive_axis(self) -> None:
        matrix = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        assert _core.sum(matrix, -1).tolist() == _core.sum(matrix, 1).tolist()
        assert _core.mean(matrix, -1, True).tolist() == _core.mean(matrix, 1, True).tolist()

    def test_axis_reductions_return_tensor_results(self) -> None:
        tensor = Tensor([2, 2, 2])

        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]):
            tensor[index] = value

        result = _core.sum(tensor, 1, True)

        assert isinstance(result, Tensor)
        assert result.shape == tensor.shape.__class__([2, 1, 2])
        assert result.tolist() == [[[4.0, 6.0]], [[12.0, 14.0]]]

    def test_tensor_negative_axis_matches_last_axis(self) -> None:
        tensor = Tensor([2, 2, 2])
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]):
            tensor[index] = value

        assert _core.sum(tensor, -1).tolist() == _core.sum(tensor, 2).tolist()
