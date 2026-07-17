from __future__ import annotations

from stratax import (
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

    def test_exceptions_use_builtins_module_name(self) -> None:
        assert StrataxError.__module__ == "builtins"
        assert ShapeError.__module__ == "builtins"
        assert DimensionError.__module__ == "builtins"
        assert IndexError.__module__ == "builtins"
        assert TypeError.__module__ == "builtins"
        assert BroadcastError.__module__ == "builtins"
        assert ZeroDivisionError.__module__ == "builtins"
