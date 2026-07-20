"""Public exception classes re-exported from the core extension."""

from ._core import (
    BroadcastError,
    DimensionError,
    IndexError,
    ShapeError,
    StrataxError,
    TypeError,
    ZeroDivisionError,
)

__all__ = [
    "StrataxError",
    "ShapeError",
    "DimensionError",
    "IndexError",
    "TypeError",
    "BroadcastError",
    "ZeroDivisionError",
]
