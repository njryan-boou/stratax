from __future__ import annotations

import stratax
from pathlib import Path


class TestPackageImports:
    def test_tests_load_the_selected_package(self, pytestconfig) -> None:
        source = Path(pytestconfig.rootpath) / "python" / "stratax"
        package = Path(stratax.__file__).resolve().parent
        extension = Path(stratax._core.__file__).resolve().parent
        assert extension == package
        if pytestconfig.getoption("--installed"):
            assert package != source
        else:
            assert package == source

    def test_public_import_surface_is_available(self) -> None:
        expected = {
            "ArrayView",
            "IndexTensor",
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
