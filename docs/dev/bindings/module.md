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
those attributes. Release version changes must update PyVersion together with
the package and Doxygen versions. Prereleases use the full Python version,
such as 0.4.0rc1, in those fields; CMake's project VERSION uses the numeric base
0.4.0. See @ref releasing for the version and artifact checks.

StrataxError derives from Python RuntimeError. All ten specialized C++ errors
are registered beneath it. Built-in conversion/slice/argument errors can also
propagate; the shared raise_overflow helper specifically uses built-in
OverflowError. See @ref python_api for the user-facing exception boundary.

Python dimensions must fit both the signed long long input conversion and the
native size_t representation. Constructors reject larger values with built-in
OverflowError before converting them; Shape metadata can represent dimensions
above the signed index limit on 32-bit builds. Container storage checks may
impose smaller limits.
