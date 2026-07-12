"""Public Stratax exception aliases exported with clean Python names."""

from ._core import (
    BroadcastError as BroadcastError,
    DimensionError as DimensionError,
    IndexError as IndexError,
    ShapeError as ShapeError,
    StrataxError as StrataxError,
    TypeError as TypeError,
    ZeroDivisionError as ZeroDivisionError,
)

__all__: list[str]
