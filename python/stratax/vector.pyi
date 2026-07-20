"""Vector type stubs for Stratax.

This module describes the typed API for ``Vector``, including construction,
indexing, arithmetic operators, reshaping, and conversion helpers.
"""

from collections.abc import Iterable, Iterator
from typing import overload

from .shape import Shape
from .tensor import Tensor


class Vector:
    """Rank-1 floating-point array with non-negative sizes."""

    @overload
    def __init__(self) -> None:
        """Create an empty vector.

        Parameters
        ----------
        self : Vector
            Target instance being initialized.

        Throws
        ------
        None

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Vector
        >>> v = Vector()
        """
        ...

    @overload
    def __init__(self, other: "Vector") -> None:
        """Copy another vector.

        Parameters
        ----------
        other : Vector
            Existing vector to copy.

        Throws
        ------
        TypeError
            If ``other`` is not a ``Vector``.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Vector
        >>> v = Vector(Vector([1.0, 2.0]))
        """
        ...

    @overload
    def __init__(self, shape: Shape) -> None:
        """Create a vector from a rank-1 shape.

        Parameters
        ----------
        shape : Shape
            Shape metadata with one dimension.

        Throws
        ------
        TypeError
            If ``shape`` is not a ``Shape``.
        ValueError
            If ``shape`` is not rank 1.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Shape, Vector
        >>> v = Vector(Shape([4]))
        """
        ...

    @overload
    def __init__(self, size: int) -> None:
        """Create a vector with ``size`` elements.

        Parameters
        ----------
        size : int
            Number of elements.

        Throws
        ------
        TypeError
            If ``size`` is not an integer.
        ValueError
            If ``size`` is negative.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Vector
        >>> v = Vector(4)
        """
        ...

    @overload
    def __init__(self, size: int, value: float) -> None:
        """Create a vector and fill with ``value``.

        Parameters
        ----------
        size : int
            Number of elements.
        value : float
            Fill value.

        Throws
        ------
        TypeError
            If ``size`` is not an integer or ``value`` is not numeric.
        ValueError
            If ``size`` is negative.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Vector
        >>> v = Vector(3, 1.5)
        """
        ...

    @overload
    def __init__(self, values: Iterable[float]) -> None:
        """Create a vector from iterable values.

        Parameters
        ----------
        values : Iterable[float]
            Sequence of numeric values.

        Throws
        ------
        TypeError
            If any item is not numeric.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Vector
        >>> v = Vector([1.0, 2.0, 3.0])
        """
        ...

    @property
    def size(self) -> int:
        """Return element count.

        Parameters
        ----------
        self : Vector
            Source vector.

        Throws
        ------
        None

        Returns
        -------
        int
            Number of elements.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0, 2.0]).size
        2
        """
        ...

    @property
    def rank(self) -> int:
        """Return tensor rank.

        Parameters
        ----------
        self : Vector
            Source vector.

        Throws
        ------
        None

        Returns
        -------
        int
            Rank value.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0]).rank
        1
        """
        ...

    @property
    def empty(self) -> bool:
        """Return whether the vector is empty.

        Parameters
        ----------
        self : Vector
            Source vector.

        Throws
        ------
        None

        Returns
        -------
        bool
            ``True`` if the vector contains no elements.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector().empty
        True
        """
        ...

    @property
    def shape(self) -> Shape:
        """Return shape metadata.

        Parameters
        ----------
        self : Vector
            Source vector.

        Throws
        ------
        None

        Returns
        -------
        Shape
            Rank-1 shape.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0, 2.0]).shape
        (2)
        """
        ...

    @property
    def strides(self) -> list[int]:
        """Return row-major strides.

        Parameters
        ----------
        self : Vector
            Source vector.

        Throws
        ------
        None

        Returns
        -------
        list[int]
            Stride metadata.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0, 2.0]).strides
        [1]
        """
        ...

    def fill(self, value: float) -> None:
        """Assign one value to all elements.

        Parameters
        ----------
        value : float
            Value to write.

        Throws
        ------
        TypeError
            If ``value`` is not numeric.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Vector
        >>> v = Vector(3)
        >>> v.fill(2.0)
        """
        ...

    def tolist(self) -> list[float]:
        """Materialize vector values as a Python list.

        Parameters
        ----------
        self : Vector
            Source vector.

        Throws
        ------
        None

        Returns
        -------
        list[float]
            Vector values in storage order.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0, 2.0]).tolist()
        [1.0, 2.0]
        """
        ...

    def reshape(self, shape: Shape | Iterable[int]) -> Tensor:
        """Return a reshaped tensor copy.

        Parameters
        ----------
        shape : Shape | Iterable[int]
            Target dimensions with the same total element count.

        Throws
        ------
        ValueError
            If element counts do not match.
        TypeError
            If ``shape`` cannot be interpreted as dimensions.

        Returns
        -------
        Tensor
            Reshaped tensor.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0, 2.0, 3.0, 4.0]).reshape([2, 2])
        """
        ...

    def flatten(self) -> "Vector":
        """Return a flattened copy.

        Parameters
        ----------
        self : Vector
            Source vector.

        Throws
        ------
        None

        Returns
        -------
        Vector
            Vector with the same values.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0, 2.0]).flatten()
        """
        ...

    def __len__(self) -> int:
        """Return element count.

        Parameters
        ----------
        self : Vector
            Source vector.

        Throws
        ------
        None

        Returns
        -------
        int
            Number of elements.

        Example
        -------
        >>> from stratax import Vector
        >>> len(Vector([1.0, 2.0]))
        2
        """
        ...

    @overload
    def __getitem__(self, index: int) -> float:
        """Read one element.

        Parameters
        ----------
        index : int
            Zero-based element index.

        Throws
        ------
        IndexError
            If ``index`` is out of range.
        TypeError
            If ``index`` is not an integer.

        Returns
        -------
        float
            Element value.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0, 2.0])[0]
        1.0
        """
        ...

    @overload
    def __getitem__(self, index: slice) -> "Vector":
        """Read a slice as a new vector.

        Parameters
        ----------
        index : slice
            Python slice bounds.

        Throws
        ------
        TypeError
            If ``index`` is not a slice.

        Returns
        -------
        Vector
            Sliced vector.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0, 2.0, 3.0])[1:]
        """
        ...

    def __setitem__(self, index: int, value: float) -> None:
        """Assign one element.

        Parameters
        ----------
        index : int
            Zero-based element index.
        value : float
            Numeric value to write.

        Throws
        ------
        IndexError
            If ``index`` is out of range.
        TypeError
            If ``index`` is not an integer or ``value`` is not numeric.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Vector
        >>> v = Vector([1.0, 2.0])
        >>> v[0] = 4.0
        """
        ...

    def __iter__(self) -> Iterator[float]:
        """Iterate over element values.

        Parameters
        ----------
        self : Vector
            Source vector.

        Throws
        ------
        None

        Returns
        -------
        Iterator[float]
            Iterator over values.

        Example
        -------
        >>> from stratax import Vector
        >>> list(Vector([1.0, 2.0]))
        [1.0, 2.0]
        """
        ...

    def __eq__(self, other: object) -> bool:
        """Compare vectors for equality.

        Parameters
        ----------
        other : object
            Value to compare.

        Throws
        ------
        None

        Returns
        -------
        bool
            ``True`` when shape and values match.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0]) == Vector([1.0])
        True
        """
        ...

    def __ne__(self, other: object) -> bool:
        """Compare vectors for inequality.

        Parameters
        ----------
        other : object
            Value to compare.

        Throws
        ------
        None

        Returns
        -------
        bool
            ``True`` when vectors differ.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0]) != Vector([2.0])
        True
        """
        ...

    def __add__(self, other: "Vector | float") -> "Vector":
        """Add vector or scalar.

        Parameters
        ----------
        other : Vector | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If vector shapes are incompatible.
        TypeError
            If ``other`` is not a supported type.

        Returns
        -------
        Vector
            Sum result.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0, 2.0]) + 1.0
        """
        ...

    def __radd__(self, other: float) -> "Vector":
        """Add scalar on the left.

        Parameters
        ----------
        other : float
            Left scalar operand.

        Throws
        ------
        TypeError
            If ``other`` is not numeric.

        Returns
        -------
        Vector
            Sum result.

        Example
        -------
        >>> from stratax import Vector
        >>> 1.0 + Vector([1.0, 2.0])
        """
        ...

    def __sub__(self, other: "Vector | float") -> "Vector":
        """Subtract vector or scalar.

        Parameters
        ----------
        other : Vector | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If vector shapes are incompatible.
        TypeError
            If ``other`` is not a supported type.

        Returns
        -------
        Vector
            Difference result.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([2.0, 3.0]) - 1.0
        """
        ...

    def __rsub__(self, other: float) -> "Vector":
        """Subtract vector from scalar.

        Parameters
        ----------
        other : float
            Left scalar operand.

        Throws
        ------
        TypeError
            If ``other`` is not numeric.

        Returns
        -------
        Vector
            Difference result.

        Example
        -------
        >>> from stratax import Vector
        >>> 5.0 - Vector([1.0, 2.0])
        """
        ...

    def __mul__(self, other: "Vector | float") -> "Vector":
        """Multiply by vector or scalar.

        Parameters
        ----------
        other : Vector | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If vector shapes are incompatible.
        TypeError
            If ``other`` is not a supported type.

        Returns
        -------
        Vector
            Product result.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([1.0, 2.0]) * 2.0
        """
        ...

    def __rmul__(self, other: float) -> "Vector":
        """Multiply scalar on the left.

        Parameters
        ----------
        other : float
            Left scalar operand.

        Throws
        ------
        TypeError
            If ``other`` is not numeric.

        Returns
        -------
        Vector
            Product result.

        Example
        -------
        >>> from stratax import Vector
        >>> 2.0 * Vector([1.0, 2.0])
        """
        ...

    def __truediv__(self, other: "Vector | float") -> "Vector":
        """Divide by vector or scalar.

        Parameters
        ----------
        other : Vector | float
            Right-hand operand.

        Throws
        ------
        ZeroDivisionError
            If any divisor is zero.
        ValueError
            If vector shapes are incompatible.

        Returns
        -------
        Vector
            Quotient result.

        Example
        -------
        >>> from stratax import Vector
        >>> Vector([2.0, 4.0]) / 2.0
        """
        ...

    def __rtruediv__(self, other: float) -> "Vector":
        """Divide scalar by vector elements.

        Parameters
        ----------
        other : float
            Left scalar operand.

        Throws
        ------
        ZeroDivisionError
            If any vector element is zero.

        Returns
        -------
        Vector
            Quotient result.

        Example
        -------
        >>> from stratax import Vector
        >>> 8.0 / Vector([2.0, 4.0])
        """
        ...

    def __iadd__(self, other: "Vector | float") -> "Vector":
        """Perform in-place addition.

        Parameters
        ----------
        other : Vector | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If vector shapes are incompatible.

        Returns
        -------
        Vector
            Updated instance.

        Example
        -------
        >>> from stratax import Vector
        >>> v = Vector([1.0, 2.0]); v += 1.0
        """
        ...

    def __isub__(self, other: "Vector | float") -> "Vector":
        """Perform in-place subtraction.

        Parameters
        ----------
        other : Vector | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If vector shapes are incompatible.

        Returns
        -------
        Vector
            Updated instance.

        Example
        -------
        >>> from stratax import Vector
        >>> v = Vector([2.0, 3.0]); v -= 1.0
        """
        ...

    def __imul__(self, other: "Vector | float") -> "Vector":
        """Perform in-place multiplication.

        Parameters
        ----------
        other : Vector | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If vector shapes are incompatible.

        Returns
        -------
        Vector
            Updated instance.

        Example
        -------
        >>> from stratax import Vector
        >>> v = Vector([1.0, 2.0]); v *= 2.0
        """
        ...

    def __itruediv__(self, other: "Vector | float") -> "Vector":
        """Perform in-place division.

        Parameters
        ----------
        other : Vector | float
            Right-hand operand.

        Throws
        ------
        ZeroDivisionError
            If any divisor is zero.
        ValueError
            If vector shapes are incompatible.

        Returns
        -------
        Vector
            Updated instance.

        Example
        -------
        >>> from stratax import Vector
        >>> v = Vector([2.0, 4.0]); v /= 2.0
        """
        ...

    def __pos__(self) -> "Vector":
        """Return a unary-plus copy.

        Parameters
        ----------
        self : Vector
            Source vector.

        Throws
        ------
        None

        Returns
        -------
        Vector
            Copy of the vector.

        Example
        -------
        >>> from stratax import Vector
        >>> +Vector([1.0, 2.0])
        """
        ...

    def __neg__(self) -> "Vector":
        """Return an elementwise negated copy.

        Parameters
        ----------
        self : Vector
            Source vector.

        Throws
        ------
        None

        Returns
        -------
        Vector
            Negated vector.

        Example
        -------
        >>> from stratax import Vector
        >>> -Vector([1.0, -2.0])
        """
        ...

    def __repr__(self) -> str:
        """Return a debug representation.

        Parameters
        ----------
        self : Vector
            Source vector.

        Throws
        ------
        None

        Returns
        -------
        str
            Printable vector representation.

        Example
        -------
        >>> from stratax import Vector
        >>> repr(Vector([1.0, 2.0]))
        '[1.0, 2.0]'
        """
        ...
