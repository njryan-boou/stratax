import pytest
import stratax as sx


@pytest.mark.parametrize("shape,index", [([], ()), ([0], (0,)), ([2, 0, 3], (0, 0, 0))])
def test_empty_tensor_checked_reads_and_writes_raise_index_error(shape, index):
    tensor = sx.Tensor(shape)
    with pytest.raises(sx.IndexError):
        tensor[index]
    with pytest.raises(sx.IndexError):
        tensor[index] = 17
    for flat_index in (0, -1):
        with pytest.raises(sx.IndexError):
            tensor[flat_index]
        with pytest.raises(sx.IndexError):
            tensor[flat_index] = 17
    assert tensor.shape == sx.Shape(shape)
    assert list(tensor) == []


@pytest.mark.parametrize("shape,index", [([], (0,)), ([0], ()), ([2, 0, 3], ())])
def test_empty_tensor_rank_errors_take_precedence(shape, index):
    tensor = sx.Tensor(shape)
    with pytest.raises(sx.RankError):
        tensor[index]
    with pytest.raises(sx.RankError):
        tensor[index] = 17


@pytest.mark.parametrize("reducer", [sx.argmin, sx.argmax])
@pytest.mark.parametrize("shape", [[0, 2], [1, 2]])
def test_index_tensor_empty_tuple_retains_rank_error(reducer, shape):
    # IndexTensor has no public constructor and axis reductions retain at least
    # one axis. Exercise both its empty and populated tuple-indexing routes.
    indices = reducer(sx.Tensor(shape, 1), axis=1)
    assert isinstance(indices, sx.IndexTensor)
    with pytest.raises(sx.RankError):
        indices[()]
    if indices.empty:
        with pytest.raises(sx.IndexError):
            indices[(0,)]
        with pytest.raises(sx.IndexError):
            indices[0]
    else:
        assert indices[(-1,)] == 0
