"""Matrix type stubs for Stratax.

This module provides static typing and documentation for ``Matrix``, a
rank-2 row-major container with slicing, element access, and arithmetic.
"""

from collections.abc import Iterable, Iterator
from typing import overload

from .shape import Shape
from .tensor import Tensor
from .vector import Vector


class Matrix:
    """Rank-2 row-major floating-point array with non-negative dimensions."""

    @overload
    def __init__(self) -> None:
        """Create an empty matrix.

        Parameters
        ----------
        self : Matrix
            Target instance being initialized.

        Throws
        ------
        None

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Matrix
        >>> m = Matrix()
        """
        ...

    @overload
    def __init__(self, other: "Matrix") -> None:
        """Copy another matrix.

        Parameters
        ----------
        other : Matrix
            Existing matrix to copy.

        Throws
        ------
        TypeError
            If ``other`` is not a ``Matrix``.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Matrix
        >>> m = Matrix(Matrix([[1.0, 2.0]]))
        """
        ...

    @overload
    def __init__(self, shape: Shape) -> None:
        """Create a matrix from rank-2 shape metadata.

        Parameters
        ----------
        shape : Shape
            Shape with two dimensions.

        Throws
        ------
        TypeError
            If ``shape`` is not a ``Shape``.
        ValueError
            If ``shape`` is not rank 2.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Matrix, Shape
        >>> m = Matrix(Shape([2, 3]))
        """
        ...

    @overload
    def __init__(self, rows: int, cols: int) -> None:
        """Create a matrix with default values.

        Parameters
        ----------
        rows : int
            Row count.
        cols : int
            Column count.

        Throws
        ------
        TypeError
            If either dimension is not an integer.
        ValueError
            If any dimension is negative.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Matrix
        >>> m = Matrix(2, 3)
        """
        ...

    @overload
    def __init__(self, rows: int, cols: int, value: float) -> None:
        """Create a matrix filled with one value.

        Parameters
        ----------
        rows : int
            Row count.
        cols : int
            Column count.
        value : float
            Fill value.

        Throws
        ------
        TypeError
            If dimensions are not integers or value is not numeric.
        ValueError
            If any dimension is negative.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Matrix
        >>> m = Matrix(2, 2, 1.0)
        """
        ...

    @overload
    def __init__(self, values: Iterable[Iterable[float]]) -> None:
        """Create a matrix from row-major nested iterables.

        Parameters
        ----------
        values : Iterable[Iterable[float]]
            Matrix rows with equal lengths.

        Throws
        ------
        TypeError
            If any value is not numeric.
        ValueError
            If rows are ragged.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Matrix
        >>> m = Matrix([[1.0, 2.0], [3.0, 4.0]])
        """
        ...

    @property
    def size(self) -> int:
        """Return total element count.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        int
            Number of elements.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix(2, 3).size
        6
        """
        ...
    @property
    def rank(self) -> int:
        """Return rank value.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        int
            Always 2 for non-empty matrices.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix(1, 1).rank
        2
        """
        ...
    @property
    def empty(self) -> bool:
        """Return whether matrix has no elements.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        bool
            ``True`` when size is 0.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix().empty
        True
        """
        ...
    @property
    def rows(self) -> int:
        """Return row count.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        int
            Number of rows.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix(2, 3).rows
        2
        """
        ...
    @property
    def cols(self) -> int:
        """Return column count.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        int
            Number of columns.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix(2, 3).cols
        3
        """
        ...
    @property
    def shape(self) -> Shape:
        """Return shape metadata.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        Shape
            Matrix shape.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix(2, 3).shape
        (2, 3)
        """
        ...
    @property
    def strides(self) -> list[int]:
        """Return row-major strides.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        list[int]
            Stride metadata.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix(2, 3).strides
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
        >>> from stratax import Matrix
        >>> m = Matrix(2, 2)
        >>> m.fill(3.0)
        """
        ...

    def tolist(self) -> list[list[float]]:
        """Materialize matrix values as nested lists.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        list[list[float]]
            Row-major values.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix([[1.0, 2.0]]).tolist()
        [[1.0, 2.0]]
        """
        ...

    def reshape(self, shape: Shape | Iterable[int]) -> Tensor:
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
        >>> from stratax import Matrix
        >>> Matrix(2, 2).reshape([4])
        """
        ...

    def flatten(self) -> Vector:
        """Return a flattened vector copy.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        Vector
            Flattened values.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix([[1.0, 2.0]]).flatten()
        """
        ...

    def __len__(self) -> int:
        """Return row count.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        int
            Number of rows.

        Example
        -------
        >>> from stratax import Matrix
        >>> len(Matrix(2, 3))
        2
        """
        ...

    @overload
    def __getitem__(self, index: tuple[int, int]) -> float:
        """Read one matrix element.

        Parameters
        ----------
        index : tuple[int, int]
            ``(row, col)`` index.

        Throws
        ------
        IndexError
            If row or column is out of range.
        TypeError
            If index format is invalid.

        Returns
        -------
        float
            Element value.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix([[1.0]])[0, 0]
        1.0
        """
        ...
    @overload
    def __getitem__(self, index: slice | tuple[int | slice, int | slice]) -> "Matrix":
        """Read a matrix slice.

        Parameters
        ----------
        index : slice | tuple[int | slice, int | slice]
            Slice expression.

        Throws
        ------
        TypeError
            If slice syntax is invalid.

        Returns
        -------
        Matrix
            Sliced result.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix([[1.0, 2.0], [3.0, 4.0]])[:, 1:]
        """
        ...

    def __setitem__(self, index: tuple[int, int], value: float) -> None:
        """Assign one matrix element.

        Parameters
        ----------
        index : tuple[int, int]
            ``(row, col)`` index.
        value : float
            Numeric value to write.

        Throws
        ------
        IndexError
            If row or column is out of range.
        TypeError
            If ``index`` or ``value`` is invalid.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Matrix
        >>> m = Matrix(1, 1)
        >>> m[0, 0] = 3.0
        """
        ...

    def __iter__(self) -> Iterator[float]:
        """Iterate in row-major order.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        Iterator[float]
            Iterator over values.

        Example
        -------
        >>> from stratax import Matrix
        >>> list(Matrix([[1.0, 2.0]]))
        [1.0, 2.0]
        """
        ...

    def __eq__(self, other: object) -> bool:
        """Compare matrices for equality.

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
        >>> from stratax import Matrix
        >>> Matrix([[1.0]]) == Matrix([[1.0]])
        True
        """
        ...

    def __ne__(self, other: object) -> bool:
        """Compare matrices for inequality.

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
            ``True`` when matrices differ.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix([[1.0]]) != Matrix([[2.0]])
        True
        """
        ...

    def __add__(self, other: "Matrix | float") -> "Matrix":
        """Add matrix or scalar.

        Parameters
        ----------
        other : Matrix | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If matrix shapes are incompatible.

        Returns
        -------
        Matrix
            Sum result.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix([[1.0]]) + 1.0
        """
        ...

    def __radd__(self, other: float) -> "Matrix":
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
        Matrix
            Sum result.

        Example
        -------
        >>> from stratax import Matrix
        >>> 1.0 + Matrix([[1.0]])
        """
        ...

    def __sub__(self, other: "Matrix | float") -> "Matrix":
        """Subtract matrix or scalar.

        Parameters
        ----------
        other : Matrix | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If matrix shapes are incompatible.

        Returns
        -------
        Matrix
            Difference result.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix([[2.0]]) - 1.0
        """
        ...

    def __rsub__(self, other: float) -> "Matrix":
        """Subtract matrix from scalar.

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
        Matrix
            Difference result.

        Example
        -------
        >>> from stratax import Matrix
        >>> 5.0 - Matrix([[1.0]])
        """
        ...

    def __mul__(self, other: "Matrix | float") -> "Matrix":
        """Multiply by matrix or scalar elementwise.

        Parameters
        ----------
        other : Matrix | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If matrix shapes are incompatible.

        Returns
        -------
        Matrix
            Product result.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix([[2.0]]) * 3.0
        """
        ...

    def __rmul__(self, other: float) -> "Matrix":
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
        Matrix
            Product result.

        Example
        -------
        >>> from stratax import Matrix
        >>> 2.0 * Matrix([[2.0]])
        """
        ...

    def __truediv__(self, other: "Matrix | float") -> "Matrix":
        """Divide by matrix or scalar elementwise.

        Parameters
        ----------
        other : Matrix | float
            Right-hand operand.

        Throws
        ------
        ZeroDivisionError
            If any divisor is zero.
        ValueError
            If matrix shapes are incompatible.

        Returns
        -------
        Matrix
            Quotient result.

        Example
        -------
        >>> from stratax import Matrix
        >>> Matrix([[4.0]]) / 2.0
        """
        ...

    def __rtruediv__(self, other: float) -> "Matrix":
        """Divide scalar by matrix elements.

        Parameters
        ----------
        other : float
            Left scalar operand.

        Throws
        ------
        ZeroDivisionError
            If any matrix element is zero.

        Returns
        -------
        Matrix
            Quotient result.

        Example
        -------
        >>> from stratax import Matrix
        >>> 8.0 / Matrix([[2.0]])
        """
        ...

    def __iadd__(self, other: "Matrix | float") -> "Matrix":
        """Perform in-place addition.

        Parameters
        ----------
        other : Matrix | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If matrix shapes are incompatible.

        Returns
        -------
        Matrix
            Updated instance.

        Example
        -------
        >>> from stratax import Matrix
        >>> m = Matrix([[1.0]]); m += 1.0
        """
        ...

    def __isub__(self, other: "Matrix | float") -> "Matrix":
        """Perform in-place subtraction.

        Parameters
        ----------
        other : Matrix | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If matrix shapes are incompatible.

        Returns
        -------
        Matrix
            Updated instance.

        Example
        -------
        >>> from stratax import Matrix
        >>> m = Matrix([[2.0]]); m -= 1.0
        """
        ...

    def __imul__(self, other: "Matrix | float") -> "Matrix":
        """Perform in-place multiplication.

        Parameters
        ----------
        other : Matrix | float
            Right-hand operand.

        Throws
        ------
        ValueError
            If matrix shapes are incompatible.

        Returns
        -------
        Matrix
            Updated instance.

        Example
        -------
        >>> from stratax import Matrix
        >>> m = Matrix([[2.0]]); m *= 2.0
        """
        ...

    def __itruediv__(self, other: "Matrix | float") -> "Matrix":
        """Perform in-place division.

        Parameters
        ----------
        other : Matrix | float
            Right-hand operand.

        Throws
        ------
        ZeroDivisionError
            If any divisor is zero.
        ValueError
            If matrix shapes are incompatible.

        Returns
        -------
        Matrix
            Updated instance.

        Example
        -------
        >>> from stratax import Matrix
        >>> m = Matrix([[4.0]]); m /= 2.0
        """
        ...

    def __pos__(self) -> "Matrix":
        """Return a unary-plus copy.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        Matrix
            Copy of matrix.

        Example
        -------
        >>> from stratax import Matrix
        >>> +Matrix([[1.0]])
        """
        ...

    def __neg__(self) -> "Matrix":
        """Return an elementwise negated copy.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        Matrix
            Negated matrix.

        Example
        -------
        >>> from stratax import Matrix
        >>> -Matrix([[1.0]])
        """
        ...

    def __repr__(self) -> str:
        """Return a debug representation.

        Parameters
        ----------
        self : Matrix
            Source matrix.

        Throws
        ------
        None

        Returns
        -------
        str
            Printable matrix representation.

        Example
        -------
        >>> from stratax import Matrix
        >>> repr(Matrix([[1.0]]))
        '[[1.0]]'
        """
        ...
