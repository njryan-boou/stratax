from __future__ import annotations

from stratax import (
    AxisError,
    BroadcastError,
    DimensionError,
    IndexError,
    ShapeError,
    StrataxError,
    TypeError,
    ZeroDivisionError,
)


class TestExceptionsInterfaceTests:
    def test_exception_hierarchy(self) -> None:
        assert issubclass(StrataxError, RuntimeError)
        assert issubclass(ShapeError, StrataxError)
        assert issubclass(DimensionError, StrataxError)
        assert issubclass(IndexError, StrataxError)
        assert issubclass(TypeError, StrataxError)
        assert issubclass(BroadcastError, StrataxError)
        assert issubclass(ZeroDivisionError, StrataxError)
        assert issubclass(AxisError, StrataxError)

    def test_exceptions_use_core_module_name(self) -> None:
        assert StrataxError.__module__ == "stratax._core"
        assert ShapeError.__module__ == "stratax._core"
        assert DimensionError.__module__ == "stratax._core"
        assert IndexError.__module__ == "stratax._core"
        assert TypeError.__module__ == "stratax._core"
        assert BroadcastError.__module__ == "stratax._core"
        assert ZeroDivisionError.__module__ == "stratax._core"
        assert AxisError.__module__ == "stratax._core"
