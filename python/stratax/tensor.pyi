"""Tensor type stubs for Stratax.

This module defines the typed API for ``Tensor``, including N-dimensional
construction, indexing, slicing, reshaping, flattening, and arithmetic.
"""

from collections.abc import Iterable, Iterator
from typing import overload

from .shape import Shape
from .vector import Vector


class Tensor:
    """N-dimensional row-major floating-point array with non-negative dimensions."""

    @overload
    def __init__(self) -> None:
        """Create an empty tensor.

        Parameters
        ----------
        self : Tensor
            Target instance being initialized.

        Throws
        ------
        None

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Tensor
        >>> t = Tensor()
        """
        ...

    @overload
    def __init__(self, other: "Tensor") -> None:
        """Copy another tensor.

        Parameters
        ----------
        other : Tensor
            Existing tensor to copy.

        Throws
        ------
        TypeError
            If ``other`` is not a ``Tensor``.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Tensor
        >>> t = Tensor(Tensor([2, 2]))
        """
        ...

    @overload
    def __init__(self, shape: Shape) -> None:
        """Create a tensor from shape metadata.

        Parameters
        ----------
        shape : Shape
            Target dimensions.

        Throws
        ------
        TypeError
            If ``shape`` is not a ``Shape``.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Shape, Tensor
        >>> t = Tensor(Shape([2, 3, 4]))
        """
        ...

    @overload
    def __init__(self, shape: Shape, value: float) -> None:
        """Create a tensor from shape and fill value.

        Parameters
        ----------
        shape : Shape
            Target dimensions.
        value : float
            Fill value.

        Throws
        ------
        TypeError
            If ``value`` is not numeric.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Shape, Tensor
        >>> t = Tensor(Shape([2, 2]), 1.0)
        """
        ...

    @overload
    def __init__(self, shape: Iterable[int]) -> None:
        """Create a tensor from iterable dimensions.

        Parameters
        ----------
        shape : Iterable[int]
            Target dimensions.

        Throws
        ------
        TypeError
            If dimensions are not integers.
        ValueError
            If any dimension is negative.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Tensor
        >>> t = Tensor([2, 2, 2])
        """
        ...

    @overload
    def __init__(self, shape: Iterable[int], value: float) -> None:
        """Create a filled tensor from iterable dimensions.

        Parameters
        ----------
        shape : Iterable[int]
            Target dimensions.
        value : float
            Fill value.

        Throws
        ------
        TypeError
            If dimensions are invalid or ``value`` is not numeric.
        ValueError
            If any dimension is negative.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Tensor
        >>> t = Tensor([2, 2], 1.0)
        """
        ...

    @property
    def size(self) -> int:
        """Return total element count.

        Parameters
        ----------
        self : Tensor
            Source tensor.

        Throws
        ------
        None

        Returns
        -------
        int
            Number of elements.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2, 3]).size
        6
        """
        ...
    @property
    def rank(self) -> int:
        """Return rank value.

        Parameters
        ----------
        self : Tensor
            Source tensor.

        Throws
        ------
        None

        Returns
        -------
        int
            Number of dimensions.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2, 3]).rank
        2
        """
        ...
    @property
    def empty(self) -> bool:
        """Return whether tensor has no elements.

        Parameters
        ----------
        self : Tensor
            Source tensor.

        Throws
        ------
        None

        Returns
        -------
        bool
            ``True`` when size is 0.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor().empty
        True
        """
        ...
    @property
    def shape(self) -> Shape:
        """Return shape metadata.

        Parameters
        ----------
        self : Tensor
            Source tensor.

        Throws
        ------
        None

        Returns
        -------
        Shape
            Tensor shape.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2, 3]).shape
        (2, 3)
        """
        ...
    @property
    def strides(self) -> list[int]:
        """Return row-major strides.

        Parameters
        ----------
        self : Tensor
            Source tensor.

        Throws
        ------
        None

        Returns
        -------
        list[int]
            Stride metadata.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2, 3]).strides
        [3, 1]
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
        >>> from stratax import Tensor
        >>> t = Tensor([2, 2])
        >>> t.fill(3.0)
        """
        ...

    def tolist(self) -> list[float]:
        """Materialize tensor values as nested Python lists.

        Parameters
        ----------
        self : Tensor
            Source tensor.

        Throws
        ------
        None

        Returns
        -------
        list[float]
            Tensor values in row-major order.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2]).tolist()
        [0.0, 0.0]
        """
        ...

    def reshape(self, shape: Shape | Iterable[int]) -> "Tensor":
        """Return a reshaped tensor copy.

        Parameters
        ----------
        shape : Shape | Iterable[int]
            Target dimensions with the same element count.

        Throws
        ------
        ValueError
            If element counts do not match.
        TypeError
            If ``shape`` is invalid.

        Returns
        -------
        Tensor
            Reshaped tensor.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2, 2]).reshape([4])
        """
        ...

    def flatten(self) -> Vector:
        """Return a flattened vector copy.

        Parameters
        ----------
        self : Tensor
            Source tensor.

        Throws
        ------
        None

        Returns
        -------
        Vector
            Flattened values.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2, 2]).flatten()
        """
        ...

    def __len__(self) -> int:
        """Return size of first axis.

        Parameters
        ----------
        self : Tensor
            Source tensor.

        Throws
        ------
        None

        Returns
        -------
        int
            Length of axis 0.

        Example
        -------
        >>> from stratax import Tensor
        >>> len(Tensor([2, 3]))
        2
        """
        ...

    @overload
    def __getitem__(self, index: int | tuple[int, ...]) -> float:
        """Read one element by integer index tuple.

        Parameters
        ----------
        index : int | tuple[int, ...]
            Index for one element.

        Throws
        ------
        IndexError
            If index is out of range.
        TypeError
            If index format is invalid.

        Returns
        -------
        float
            Element value.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2])[0]
        0.0
        """
        ...
    @overload
    def __getitem__(self, index: slice | tuple[int | slice, ...]) -> "Tensor":
        """Read a slice as a tensor.

        Parameters
        ----------
        index : slice | tuple[int | slice, ...]
            Slice expression.

        Throws
        ------
        TypeError
            If index format is invalid.

        Returns
        -------
        Tensor
            Sliced result.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([4])[1:3]
        """
        ...

    def __setitem__(self, index: int | tuple[int, ...], value: float) -> None:
        """Assign one element.

        Parameters
        ----------
        index : int | tuple[int, ...]
            Index for one element.
        value : float
            Numeric value to write.

        Throws
        ------
        IndexError
            If index is out of range.
        TypeError
            If index format or value type is invalid.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Tensor
        >>> t = Tensor([2]); t[0] = 1.0
        """
        ...

    def __iter__(self) -> Iterator[float]:
        """Iterate in row-major order.

        Parameters
        ----------
        self : Tensor
            Source tensor.

        Throws
        ------
        None

        Returns
        -------
        Iterator[float]
            Iterator over flattened values.

        Example
        -------
        >>> from stratax import Tensor
        >>> list(Tensor([2]))
        [0.0, 0.0]
        """
        ...

    def __eq__(self, other: object) -> bool:
        """Compare tensors for equality.

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
        >>> from stratax import Tensor
        >>> Tensor([2]) == Tensor([2])
        True
        """
        ...

    def __ne__(self, other: object) -> bool:
        """Compare tensors for inequality.

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
            ``True`` when tensors differ.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2]) != Tensor([3])
        True
        """
        ...

    def __add__(self, other: "Tensor | float") -> "Tensor":
        """Add tensor or scalar elementwise.

        Parameters
        ----------
        other : Tensor | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If tensor shapes are incompatible.

        Returns
        -------
        Tensor
            Sum result.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2]) + 1.0
        """
        ...

    def __radd__(self, other: float) -> "Tensor":
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
        Tensor
            Sum result.

        Example
        -------
        >>> from stratax import Tensor
        >>> 1.0 + Tensor([2])
        """
        ...

    def __sub__(self, other: "Tensor | float") -> "Tensor":
        """Subtract tensor or scalar elementwise.

        Parameters
        ----------
        other : Tensor | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If tensor shapes are incompatible.

        Returns
        -------
        Tensor
            Difference result.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2]) - 1.0
        """
        ...

    def __rsub__(self, other: float) -> "Tensor":
        """Subtract tensor from scalar.

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
        Tensor
            Difference result.

        Example
        -------
        >>> from stratax import Tensor
        >>> 5.0 - Tensor([2])
        """
        ...

    def __mul__(self, other: "Tensor | float") -> "Tensor":
        """Multiply tensor or scalar elementwise.

        Parameters
        ----------
        other : Tensor | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If tensor shapes are incompatible.

        Returns
        -------
        Tensor
            Product result.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2]) * 2.0
        """
        ...

    def __rmul__(self, other: float) -> "Tensor":
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
        Tensor
            Product result.

        Example
        -------
        >>> from stratax import Tensor
        >>> 2.0 * Tensor([2])
        """
        ...

    def __truediv__(self, other: "Tensor | float") -> "Tensor":
        """Divide by tensor or scalar elementwise.

        Parameters
        ----------
        other : Tensor | float
            Right-hand operand.

        Throws
        ------
        ZeroDivisionError
            If any divisor is zero.
        ValueError
            If tensor shapes are incompatible.

        Returns
        -------
        Tensor
            Quotient result.

        Example
        -------
        >>> from stratax import Tensor
        >>> Tensor([2]) / 2.0
        """
        ...

    def __rtruediv__(self, other: float) -> "Tensor":
        """Divide scalar by tensor elements.

        Parameters
        ----------
        other : float
            Left scalar operand.

        Throws
        ------
        ZeroDivisionError
            If any tensor element is zero.

        Returns
        -------
        Tensor
            Quotient result.

        Example
        -------
        >>> from stratax import Tensor
        >>> 8.0 / Tensor([2])
        """
        ...

    def __iadd__(self, other: "Tensor | float") -> "Tensor":
        """Perform in-place addition.

        Parameters
        ----------
        other : Tensor | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If tensor shapes are incompatible.

        Returns
        -------
        Tensor
            Updated instance.

        Example
        -------
        >>> from stratax import Tensor
        >>> t = Tensor([2]); t += 1.0
        """
        ...

    def __isub__(self, other: "Tensor | float") -> "Tensor":
        """Perform in-place subtraction.

        Parameters
        ----------
        other : Tensor | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If tensor shapes are incompatible.

        Returns
        -------
        Tensor
            Updated instance.

        Example
        -------
        >>> from stratax import Tensor
        >>> t = Tensor([2]); t -= 1.0
        """
        ...

    def __imul__(self, other: "Tensor | float") -> "Tensor":
        """Perform in-place multiplication.

        Parameters
        ----------
        other : Tensor | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If tensor shapes are incompatible.

        Returns
        -------
        Tensor
            Updated instance.

        Example
        -------
        >>> from stratax import Tensor
        >>> t = Tensor([2]); t *= 2.0
        """
        ...

    def __itruediv__(self, other: "Tensor | float") -> "Tensor":
        """Perform in-place division.

        Parameters
        ----------
        other : Tensor | float
            Right-hand operand.

        Throws
        ------
        ZeroDivisionError
            If any divisor is zero.
        ValueError
            If tensor shapes are incompatible.

        Returns
        -------
        Tensor
            Updated instance.

        Example
        -------
        >>> from stratax import Tensor
        >>> t = Tensor([2]); t /= 2.0
        """
        ...

    def __pos__(self) -> "Tensor":
        """Return a unary-plus copy.

        Parameters
        ----------
        self : Tensor
            Source tensor.

        Throws
        ------
        None

        Returns
        -------
        Tensor
            Copy of tensor.

        Example
        -------
        >>> from stratax import Tensor
        >>> +Tensor([2])
        """
        ...

    def __neg__(self) -> "Tensor":
        """Return an elementwise negated copy.

        Parameters
        ----------
        self : Tensor
            Source tensor.

        Throws
        ------
        None

        Returns
        -------
        Tensor
            Negated tensor.

        Example
        -------
        >>> from stratax import Tensor
        >>> -Tensor([2])
        """
        ...

    def __repr__(self) -> str:
        """Return a debug representation.

        Parameters
        ----------
        self : Tensor
            Source tensor.

        Throws
        ------
        None

        Returns
        -------
        str
            Printable tensor representation.

        Example
        -------
        >>> from stratax import Tensor
        >>> repr(Tensor([2]))
        'Tensor(shape=(2,))'
        """
        ...
