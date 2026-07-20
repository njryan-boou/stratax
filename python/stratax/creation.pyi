"""Tensor creation helpers.

These factory functions create ``Tensor`` instances with common initialization
patterns such as zeros, ones, constant fills, and identity matrices.
"""

from collections.abc import Iterable

from .shape import Shape
from .tensor import Tensor


def zeros(shape: Shape | Iterable[int]) -> Tensor:
	"""Create a zero-initialized tensor.

	Parameters
	----------
	shape : Shape | Iterable[int]
		Target dimensions.

	Throws
	------
	TypeError
		If ``shape`` cannot be interpreted as dimensions.
	ValueError
		If any dimension is negative.

	Returns
	-------
	Tensor
		Tensor filled with ``0.0``.

	Example
	-------
	>>> from stratax import zeros
	>>> zeros([2, 3])
	"""
	...


def ones(shape: Shape | Iterable[int]) -> Tensor:
	"""Create a one-initialized tensor.

	Parameters
	----------
	shape : Shape | Iterable[int]
		Target dimensions.

	Throws
	------
	TypeError
		If ``shape`` cannot be interpreted as dimensions.
	ValueError
		If any dimension is negative.

	Returns
	-------
	Tensor
		Tensor filled with ``1.0``.

	Example
	-------
	>>> from stratax import ones
	>>> ones([2, 3])
	"""
	...


def full(shape: Shape | Iterable[int], value: float) -> Tensor:
	"""Create a tensor filled with a constant value.

	Parameters
	----------
	shape : Shape | Iterable[int]
		Target dimensions.
	value : float
		Fill value.

	Throws
	------
	TypeError
		If ``shape`` is invalid or ``value`` is not numeric.
	ValueError
		If any dimension is negative.

	Returns
	-------
	Tensor
		Tensor filled with ``value``.

	Example
	-------
	>>> from stratax import full
	>>> full([2, 2], 3.5)
	"""
	...


def identity(size: int) -> Tensor:
	"""Create a 2D identity tensor.

	Parameters
	----------
	size : int
		Side length of the square identity.

	Throws
	------
	TypeError
		If ``size`` is not an integer.
	ValueError
		If ``size`` is negative.

	Returns
	-------
	Tensor
		Tensor with ones on the diagonal and zeros elsewhere.

	Example
	-------
	>>> from stratax import identity
	>>> identity(3)
	"""
	...
