from __future__ import annotations

import sys
from importlib.machinery import PathFinder
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in start.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "python" / "stratax" / "__init__.py").is_file():
            return candidate

    raise RuntimeError("Unable to locate repository root from tests/python")


ROOT = _find_repo_root(Path(__file__).resolve())


def pytest_addoption(parser):
    parser.addoption("--installed", action="store_true", help="Test the installed package instead of the source-tree extension")


def pytest_configure(config):
    if not config.getoption("--installed"):
        sys.path.insert(0, str(ROOT / "python"))
        # Editable-install finders can precede sys.path and silently load an
        # older installed extension. Source tests must use the local build.
        class SourcePackageFinder:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "stratax":
                    return PathFinder.find_spec(fullname, [str(ROOT / "python")])
                if fullname.startswith("stratax."):
                    spec = PathFinder.find_spec(fullname, path)
                    if spec is None:
                        raise ModuleNotFoundError(
                            f"{fullname} is missing from the source tree; build the Python extension first"
                        )
                    return spec
                return None

        sys.meta_path.insert(0, SourcePackageFinder())
