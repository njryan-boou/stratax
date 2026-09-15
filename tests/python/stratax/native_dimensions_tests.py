"""Dimensions must not wrap to zero or a small allocation on narrower builds."""

import ctypes

import pytest

from stratax import Matrix, Shape, Tensor, Vector


@pytest.mark.parametrize("value", [1 << (8 * ctypes.sizeof(ctypes.c_size_t)),
                                 (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) + 1])
@pytest.mark.parametrize("construct", [
    pytest.param(Shape, id="shape-scalar"),
    pytest.param(lambda n: Shape([n]), id="shape-iterable"),
    pytest.param(Vector, id="vector-size"),
    pytest.param(lambda n: Vector(n, 1.0), id="vector-fill"),
    pytest.param(lambda n: Matrix(n, 0), id="matrix-rows"),
    pytest.param(lambda n: Matrix(0, n), id="matrix-cols"),
    pytest.param(lambda n: Matrix(n, 0, 1.0), id="matrix-fill-rows"),
    pytest.param(lambda n: Matrix(0, n, 1.0), id="matrix-fill-cols"),
    pytest.param(lambda n: Tensor([n, 0]), id="tensor-first-dimension"),
    pytest.param(lambda n: Tensor([0, n]), id="tensor-after-zero"),
    pytest.param(lambda n: Tensor([n, 0], 1.0), id="tensor-fill-first-dimension"),
    pytest.param(lambda n: Tensor([0, n], 1.0), id="tensor-fill-after-zero"),
])
def test_out_of_range_dimensions_raise_before_narrowing(construct, value):
    with pytest.raises(OverflowError):
        construct(value)
