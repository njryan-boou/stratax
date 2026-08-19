from __future__ import annotations

import stratax


class TestPackageImports:
    def test_public_import_surface_is_available(self) -> None:
        expected = {
            "Shape",
            "Tensor",
            "Vector",
            "Matrix",
            "BoolVector",
            "BoolMatrix",
            "BoolTensor",
            "equal",
            "not_equal",
            "less",
            "less_equal",
            "greater",
            "greater_equal",
            "to_vector",
            "to_matrix",
            "to_tensor",
            "zeros",
            "ones",
            "full",
            "identity",
            "sum",
            "prod",
            "max",
            "min",
            "argmax",
            "argmin",
            "mean",
            "var",
            "std",
            "StrataxError",
            "ShapeError",
            "DimensionError",
            "IndexError",
            "TypeError",
            "BroadcastError",
            "ZeroDivisionError",
            "AxisError",
            "__version__",
            "__author__",
        }

        assert expected.issubset(set(stratax.__all__))

        for name in expected:
            assert getattr(stratax, name) is not None
