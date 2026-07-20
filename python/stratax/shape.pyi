"""Shape type stubs for Stratax.

This module defines the public static typing surface for ``Shape``, which
stores non-negative dimensions used across vector, matrix, and tensor APIs.
"""

from collections.abc import Iterable, Iterator
from typing import overload


class Shape:
    """Stores non-negative dimensions for a Stratax array."""

    @overload
    def __init__(self) -> None:
        """Create an empty shape.

        Parameters
        ----------
        self : Shape
            Target instance being initialized.

        Throws
        ------
        None

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Shape
        >>> s = Shape()
        """
        ...

    @overload
    def __init__(self, other: "Shape") -> None:
        """Copy another shape.

        Parameters
        ----------
        other : Shape
            Existing shape to copy.

        Throws
        ------
        TypeError
            If ``other`` is not a ``Shape``.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Shape
        >>> s = Shape(Shape([2, 3]))
        """
        ...

    @overload
    def __init__(self, dims: Iterable[int]) -> None:
        """Create a shape from an iterable of dimensions.

        Parameters
        ----------
        dims : Iterable[int]
            Dimension values in order.

        Throws
        ------
        TypeError
            If any dimension is not an integer.
        ValueError
            If any dimension is negative.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Shape
        >>> s = Shape([2, 3, 4])
        """
        ...

    @overload
    def __init__(self, first: int, second: int, /, *rest: int) -> None:
        """Create a shape from variadic dimensions.

        Parameters
        ----------
        first : int
            First dimension.
        second : int
            Second dimension.
        *rest : int
            Optional remaining dimensions.

        Throws
        ------
        TypeError
            If any dimension is not an integer.
        ValueError
            If any dimension is negative.

        Returns
        -------
        None

        Example
        -------
        >>> from stratax import Shape
        >>> s = Shape(2, 3, 4)
        """
        ...

    @property
    def rank(self) -> int:
        """Return the number of dimensions.

        Parameters
        ----------
        self : Shape
            Source shape.

        Throws
        ------
        None

        Returns
        -------
        int
            Dimension count.

        Example
        -------
        >>> from stratax import Shape
        >>> Shape([2, 3]).rank
        2
        """
        ...

    @property
    def elements(self) -> int:
        """Return the total element count implied by dimensions.

        Parameters
        ----------
        self : Shape
            Source shape.

        Throws
        ------
        None

        Returns
        -------
        int
            Product of dimensions.

        Example
        -------
        >>> from stratax import Shape
        >>> Shape([2, 3, 4]).elements
        24
        """
        ...

    @property
    def empty(self) -> bool:
        """Return whether the shape has rank 0.

        Parameters
        ----------
        self : Shape
            Source shape.

        Throws
        ------
        None

        Returns
        -------
        bool
            ``True`` if the shape has no dimensions.

        Example
        -------
        >>> from stratax import Shape
        >>> Shape().empty
        True
        """
        ...

    def __len__(self) -> int:
        """Return the rank.

        Parameters
        ----------
        self : Shape
            Source shape.

        Throws
        ------
        None

        Returns
        -------
        int
            Dimension count.

        Example
        -------
        >>> from stratax import Shape
        >>> len(Shape([2, 3]))
        2
        """
        ...

    def __getitem__(self, index: int) -> int:
        """Return a dimension by index.

        Parameters
        ----------
        index : int
            Zero-based dimension index.

        Throws
        ------
        IndexError
            If ``index`` is out of range.
        TypeError
            If ``index`` is not an integer.

        Returns
        -------
        int
            Dimension value at ``index``.

        Example
        -------
        >>> from stratax import Shape
        >>> Shape([2, 3])[1]
        3
        """
        ...

    def __iter__(self) -> Iterator[int]:
        """Iterate over dimensions.

        Parameters
        ----------
        self : Shape
            Source shape.

        Throws
        ------
        None

        Returns
        -------
        Iterator[int]
            Iterator of dimensions.

        Example
        -------
        >>> from stratax import Shape
        >>> list(Shape([2, 3]))
        [2, 3]
        """
        ...

    def __eq__(self, other: object) -> bool:
        """Compare shapes for equality.

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
            ``True`` when dimensions match exactly.

        Example
        -------
        >>> from stratax import Shape
        >>> Shape([2, 3]) == Shape([2, 3])
        True
        """
        ...

    def __ne__(self, other: object) -> bool:
        """Compare shapes for inequality.

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
            ``True`` when dimensions differ.

        Example
        -------
        >>> from stratax import Shape
        >>> Shape([2, 3]) != Shape([3, 2])
        True
        """
        ...

    def __repr__(self) -> str:
        """Return a debug representation.

        Parameters
        ----------
        self : Shape
            Source shape.

        Throws
        ------
        None

        Returns
        -------
        str
            String representation of dimensions.

        Example
        -------
        >>> from stratax import Shape
        >>> repr(Shape([2, 3]))
        '(2, 3)'
        """
        ...
