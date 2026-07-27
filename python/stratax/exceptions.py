from ._core import (
    BroadcastError,
    DimensionError,
    IndexError,
    ShapeError,
    StrataxError,
    TypeError,
    ZeroDivisionError,
    AxisError
)

for _exception in (
    StrataxError,
    ShapeError,
    DimensionError,
    IndexError,
    TypeError,
    BroadcastError,
    ZeroDivisionError,
    AxisError
):
    _exception.__module__ = "builtins"

del _exception

__all__ = [
    "StrataxError",
    "ShapeError",
    "DimensionError",
    "IndexError",
    "TypeError",
    "BroadcastError",
    "ZeroDivisionError",
    "AxisError"
]
