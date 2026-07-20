"""Container conversion helpers.

These helpers adapt between ``Vector``, ``Matrix``, and ``Tensor`` wrappers
using the same conversion rules implemented by the underlying core extension.
"""

from .matrix import Matrix
from .tensor import Tensor
from .vector import Vector


def to_vector(arr: Vector | Matrix | Tensor) -> Vector:
	"""Convert a container to ``Vector``.

	Parameters
	----------
	arr : Vector | Matrix | Tensor
		Source container.

	Throws
	------
	ValueError
		If conversion is not shape-compatible.
	TypeError
		If ``arr`` is not a supported type.

	Returns
	-------
	Vector
		Converted vector.

	Example
	-------
	>>> from stratax import Matrix, to_vector
	>>> to_vector(Matrix([[1.0, 2.0]]))
	"""
	...


def to_matrix(arr: Vector | Matrix | Tensor) -> Matrix:
	"""Convert a container to ``Matrix``.

	Parameters
	----------
	arr : Vector | Matrix | Tensor
		Source container.

	Throws
	------
	ValueError
		If conversion is not shape-compatible.
	TypeError
		If ``arr`` is not a supported type.

	Returns
	-------
	Matrix
		Converted matrix.

	Example
	-------
	>>> from stratax import Tensor, to_matrix
	>>> to_matrix(Tensor([2, 2]))
	"""
	...


def to_tensor(arr: Vector | Matrix | Tensor) -> Tensor:
	"""Convert a container to ``Tensor``.

	Parameters
	----------
	arr : Vector | Matrix | Tensor
		Source container.

	Throws
	------
	TypeError
		If ``arr`` is not a supported type.

	Returns
	-------
	Tensor
		Converted tensor.

	Example
	-------
	>>> from stratax import Vector, to_tensor
	>>> to_tensor(Vector([1.0, 2.0]))
	"""
	...
