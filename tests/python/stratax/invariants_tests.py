import itertools
import math
import operator

import pytest
import stratax as sx


def tensor(shape, values):
    result = sx.Tensor(list(shape))
    for i, value in enumerate(values):
        result[i] = value
    return result


@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv, operator.eq, operator.lt])
def test_broadcasting_matches_coordinate_reference(op):
    lhs = tensor((2, 1, 3), range(1, 7))
    rhs = tensor((1, 4, 1), range(2, 6))
    for left, right in ((lhs, rhs), (rhs, lhs)):
        result = op(left, right)
        assert result.shape == sx.Shape([2, 4, 3])
        expected = [op(left[i if left is lhs else 0, 0 if left is lhs else j, k if left is lhs else 0],
                       right[0 if right is rhs else i, j if right is rhs else 0, 0 if right is rhs else k])
                    for i, j, k in itertools.product(range(2), range(4), range(3))]
        assert list(result) == pytest.approx(expected)
    assert list(lhs) == list(range(1, 7))
    assert list(rhs) == list(range(2, 6))


@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv, operator.eq, operator.lt])
def test_mixed_container_broadcasts_and_empty_results(op):
    matrix = sx.Matrix([[2, 4, 6], [8, 10, 12]])
    row = sx.Vector([1, 2, 3])
    result = op(matrix, row)
    assert result.shape == sx.Shape([2, 3])
    assert list(result) == pytest.approx([op(value, (i % 3) + 1) for i, value in enumerate(matrix)])
    assert op(sx.Tensor([0, 3]), row).shape == sx.Shape([0, 3])
    with pytest.raises(sx.BroadcastError):
        op(matrix, sx.Vector([1, 2]))
    with pytest.raises(sx.BroadcastError):
        op(sx.Tensor([]), row)
    with pytest.raises(sx.BroadcastError):
        op(row, sx.Tensor([]))


@pytest.mark.parametrize("op", [operator.iadd, operator.isub, operator.imul, operator.itruediv])
def test_inplace_broadcast_preserves_identity_and_shape(op):
    matrix = sx.Matrix([[2, 4, 6], [8, 10, 12]])
    row = sx.Vector([1, 2, 3])
    result = op(matrix, row)
    assert result is matrix
    assert matrix.shape == sx.Shape([2, 3])
    assert list(matrix) == pytest.approx([op(value, (i % 3) + 1) for i, value in enumerate([2, 4, 6, 8, 10, 12])])
    before = list(row)
    with pytest.raises(sx.BroadcastError):
        op(row, matrix)
    assert list(row) == before
    before_matrix = list(matrix)
    with pytest.raises(sx.BroadcastError):
        op(matrix, sx.Tensor([]))
    assert list(matrix) == before_matrix


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, 0, 3), (2, 3, 0)])
def test_matrix_conversion_never_drops_zero_axes(shape):
    with pytest.raises(sx.ShapeError):
        sx.to_matrix(sx.Tensor(shape))


@pytest.mark.parametrize("shape,expected", [((1, 0, 3), (0, 3)), ((2, 1, 0), (2, 0)), ((1, 0, 0), (0, 0))])
def test_matrix_conversion_removes_only_singleton_axes(shape, expected):
    result = sx.to_matrix(sx.Tensor(shape))
    assert result.shape == sx.Shape(expected)
    assert result.size == 0


@pytest.mark.parametrize("name", ["sum", "prod", "min", "max", "argmin", "argmax", "mean", "var", "std"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("keepdims", [False, True])
def test_all_axis_reductions_against_independent_reference(name, axis, keepdims):
    shape = (2, 3, 4)
    # Repeated extrema exercise first-index tie breaking.
    values = [(i * 7) % 11 + 1 for i in range(math.prod(shape))]
    source = tensor(shape, values)
    func = getattr(sx, name)
    result = func(source, axis=axis, keepdims=keepdims)
    if name in ("argmin", "argmax"):
        assert isinstance(result, sx.IndexTensor)
        assert all(type(value) is int for value in result)
    output_shape = [1 if i == axis else extent for i, extent in enumerate(shape)]
    if not keepdims:
        del output_shape[axis]
    assert result.shape == sx.Shape(output_shape)
    expected = []
    for output in itertools.product(*(range(n) for n in output_shape)):
        coordinate = list(output)
        if not keepdims:
            coordinate.insert(axis, 0)
        group = []
        for position in range(shape[axis]):
            coordinate[axis] = position
            i, j, k = coordinate
            group.append(values[(i * shape[1] + j) * shape[2] + k])
        mean = sum(group) / len(group)
        variance = sum((value - mean) ** 2 for value in group) / len(group)
        expected.append({"sum": sum(group), "prod": math.prod(group), "min": min(group), "max": max(group),
                         "argmin": group.index(min(group)), "argmax": group.index(max(group)),
                         "mean": mean, "var": variance, "std": math.sqrt(variance)}[name])
    assert list(result) == pytest.approx(expected)
    assert list(func(source, axis=axis - len(shape), keepdims=keepdims)) == pytest.approx(expected)
    assert list(source) == values


@pytest.mark.parametrize("name", ["sum", "prod", "min", "max", "argmin", "argmax", "mean", "var", "std"])
def test_reduction_invalid_axes_and_empty_output(name):
    func = getattr(sx, name)
    for axis in (-3, 2):
        with pytest.raises(sx.AxisError):
            func(sx.Tensor([2, 3]), axis=axis)
    assert func(sx.Tensor([0, 3]), axis=1).shape == sx.Shape([0])
    assert func(sx.Tensor([0, 3]), axis=1, keepdims=True).shape == sx.Shape([0, 1])
    if name in ("sum", "prod"):
        identity = 0 if name == "sum" else 1
        assert func(sx.Vector()) == identity
        assert list(func(sx.Tensor([0, 3]), axis=0)) == [identity] * 3
    else:
        error = sx.ZeroDivisionError if name in ("mean", "var", "std") else sx.IndexError
        with pytest.raises(error):
            func(sx.Vector())
        with pytest.raises(error):
            func(sx.Tensor([0, 3]), axis=0)


def test_inplace_division_failure_is_atomic():
    source = sx.Matrix([[10, 20], [30, 40]])
    before = list(source)
    with pytest.raises(sx.ZeroDivisionError):
        source /= sx.Vector([2, 0])
    assert list(source) == before
    assert source.shape == sx.Shape([2, 2])


def test_index_tensor_preserves_integer_access_and_iterator_lifetime():
    result = sx.argmax(sx.Matrix([[2, 5, 5], [7, 3, 1]]), axis=1, keepdims=True)
    assert result.tolist() == [[1], [0]]
    assert result[0, 0] == result[-2] == 1
    assert type(result[0, 0]) is int
    with pytest.raises(sx.IndexError):
        _ = result[2]
    with pytest.raises(sx.RankError):
        _ = result[0, 0, 0]
    iterator = iter(result)
    del result
    assert list(iterator) == [1, 0]
