# Changelog {#changelog}

<!-- markdownlint-disable MD024 -->

All notable changes to Stratax will be documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning while under active 0.x development.

---

## [0.2.0] - 07-15-2026

### Added

- Completed the Python API for Shape, Vector, Matrix, and Tensor.
- Added reshape and flatten methods.
- Added container conversion methods.
- Added Python creation functions.
- Added negative indexing.
- Added slicing to the Python API.
- Added complete Python type stubs.
- Expanded Python tests.

### Changed

- Reorganized the Python package into a cleaner module structure.
- Improved Python bindings for consistency with the C++ API.
- Improved conversion behavior for vector-shaped arrays.
- Improved slicing behavior to better match Python semantics.

### Fixed

- Fixed numerous Python binding issues.
- Fixed conversion edge cases for row and column vectors.
- Fixed editable install and build workflow issues.
- Improved exception translation.

---

## [0.1.0] - 07-11-2026

### Added

- Initial public release.
- Shape
- Vector
- Matrix
- Tensor
- Element-wise arithmetic
- Element-wise comparisons
- Python bindings
- Type hints
- Examples
- Doxygen documentation
