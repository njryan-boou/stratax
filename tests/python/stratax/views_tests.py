import gc
import itertools
import sys

import pytest

from stratax import ArrayView, Matrix, Tensor, Vector, Shape, RankError
from stratax import IndexError as StrataxIndexError, TypeError as StrataxTypeError


@pytest.mark.parametrize("factory", [lambda: Vector([1, 2, 3, 4]), lambda: Matrix([[1, 2], [3, 4]]), lambda: Tensor([2, 2], 1)])
def test_views_share_writes_and_keep_owner_alive(factory):
    owner = factory()
    view = owner[:] if isinstance(owner, Vector) else owner[:, :]
    assert isinstance(view, ArrayView)
    assert view.shape == owner.shape
    assert view.strides == owner.strides
    assert view.rank == view.ndim == owner.rank
    assert view.size == owner.size
    view[0] = 42
    assert owner[0] == 42
    owner[(0, 1) if isinstance(owner, Matrix) else 1] = 17
    assert view[1] == 17
    owner += 1
    assert view[0] == 43
    del owner
    gc.collect()
    assert view[0] == 43
    view[-1] = 99
    assert view[-1] == 99


def test_strided_view_indexing_and_validation():
    owner = Matrix([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    view = owner[::2, 1::2]
    assert view.shape == Shape([2, 2])
    assert view.strides == Shape([8, 2])
    assert view.tolist() == [[1, 3], [9, 11]]
    assert view[-1, -2] == view[2] == 9
    view[-1, -2] = 90
    assert owner[2, 1] == 90
    for index in (4, -5, (2, 0), (0, -3)):
        with pytest.raises(StrataxIndexError):
            _ = view[index]
        with pytest.raises(StrataxIndexError):
            view[index] = 2
    with pytest.raises(RankError):
        _ = view[0, 0, 0]
    for index in ("0", 1.5, (0, "1")):
        with pytest.raises(StrataxTypeError):
            _ = view[index]
    with pytest.raises(StrataxTypeError):
        view[0] = "bad"
    assert owner[0, 1] == 1


def test_positive_slices_match_python_lists_exhaustively():
    bounds = (None, -10, -2, -1, 0, 1, 3, 10)
    for size in range(7):
        values = list(range(size))
        vector = Vector(values)
        for start, stop, step in itertools.product(bounds, bounds, (1, 2, 3, sys.maxsize)):
            selection = slice(start, stop, step)
            expected = values[selection]
            view = vector[selection]
            assert view.tolist() == expected, (size, selection)
            assert view.size == len(expected)
            assert view.shape == Shape([len(expected)])
            assert view.empty == (not expected)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0), (2, 3)])
def test_empty_multidimensional_slices(shape):
    matrix = Matrix(*shape)
    view = matrix[shape[0]:, shape[1]:]
    assert view.empty
    assert view.shape == Shape([0, 0])
    assert view.tolist() == []
    with pytest.raises(StrataxIndexError):
        _ = view[0]


@pytest.mark.parametrize("factory", [lambda: Vector([1]), lambda: Matrix([[1]]), lambda: Tensor([1], 1)])
def test_slice_step_zero_is_rejected(factory):
    with pytest.raises(ValueError):
        _ = factory()[::0]
