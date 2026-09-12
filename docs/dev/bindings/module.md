@page dev_module _core Extension Module

# _core Extension Module

Source: `bindings/module.cpp`.

The pybind11 entry point registers exceptions, metadata, classes, and free
functions. python/stratax/__init__.py defines the public package export list.

## Registration order

1. Exceptions and Shape.
2. Boolean result classes.
3. Vector, Matrix, Tensor.
4. ArrayView and named comparisons.
5. Conversions and creation helpers.
6. Reductions, including IndexTensor registration.

Shared helpers may register operations whose return types are registered later
in initialization. All registration completes before callers invoke those
operations. Numeric bindings use double; Boolean masks and index results use
bool and dtype::int64 respectively.

## Metadata and exceptions

PyDoc, PyVersion, PyAuthor, PyLicense, and PyModule constants set __doc__,
__version__, __author__, __license__, and __module__. Homepage/repository/issue
URLs belong to package metadata in pyproject.toml; module.cpp does not register
those attributes. Release version changes must update PyVersion as well as
package/CMake/Doxygen versions.

StrataxError derives from Python RuntimeError. All ten specialized C++ errors
are registered beneath it. Built-in conversion/slice/argument errors can also
propagate; the shared raise_overflow helper specifically uses built-in
OverflowError. See @ref python_api for the user-facing exception boundary.
