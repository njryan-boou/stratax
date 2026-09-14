@page changelog Changelog

# Changelog

<!-- markdownlint-disable MD024 -->

All notable changes to Stratax will be documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning while under active 0.x development.

---

## [Unreleased]

---

## [0.4.0rc1] - 2026-09-14

Release candidate notes prepared for 0.4.0; publication is pending.

### Fixed

- Made NaN and signed-zero extrema selection consistent across compilers.
- Rejected checked multidimensional access to empty storage, including rank-zero arrays.
- Gave overlapping C++ compound operations snapshot semantics, including aliased scalars and repeated view offsets.
- Rejected unrepresentable signed division and accumulated means in long double before returning double.
- Made owning array copy assignment transactional and preserved Vector/Matrix rank after moves.
- Corrected Buffer construction cleanup, extreme Slice arithmetic, empty views, zero-axis conversions, and empty broadcasting.
- Validated compound divisors before mutation and registered Python axis-index results as IndexTensor.
- Corrected Doxygen contracts, current C++ namespaces, view semantics, dtype/result descriptions, and stale guide examples.

### Changed

- Vector and Matrix moves may allocate empty metadata and throw; Buffer/Tensor moves remain noexcept.
- Python tests distinguish source and installed extensions; CI runs package tests and Clang sanitizer checks.

### Added

- Deterministic aliasing and numerical edge tests, branch coverage reports, and reproducible performance baselines.
- CI gates for independent headers, executable documentation examples, and Doxygen warnings.
- Isolated tests of each release wheel and of packages built from the source archive.
- Boundary and invariant regression coverage, a view developer reference, and an executable documentation-example check.
- Release checks for version agreement, package metadata, runtime versions, and bundled typing files.

---

## [0.3.1] - 08-09-2026

### Added

- Added cross-platform CI coverage for Linux (GCC/Clang), Windows (MSVC), and macOS (Apple Clang), plus a Python version matrix through 3.14.
- Added cross-platform release wheel builds for Linux, Windows, and macOS via `cibuildwheel`.

### Changed

- Improved release automation by publishing combined wheel and sdist artifacts through the PyPI release workflow.

---

## [0.3.0] - 08-02-2026

### Added

- Added NumPy-style broadcasting to C++ array arithmetic, including shape validation, result-shape calculation, singleton-dimension projection, scalar operations, and compound assignment.
- Added complete reduction APIs for C++ and Python: `sum`, `prod`, `min`, `max`, `argmin`, `argmax`, `mean`, `var`, and `std`, with axis, negative-axis, and `keepdims` support.
- Added focused broadcasting tests and developer documentation alongside same-shape arithmetic coverage.
- Added a NumPy-style C++ public facade with top-level aliases and module-style namespaces.
- Added GitHub Pages documentation deployment and expanded contributor and architecture documentation.

### Changed

- Simplified Python packaging around direct exports from the compiled `_core` extension and a single public type stub.
- Updated Python API stubs to match the property-based container interface.
- Refactored the C++ and Python implementation structure to reduce duplication and keep bindings aligned with the core API.
- Refreshed the project roadmap to reflect the implemented broadcasting and reduction systems and the remaining release work.
- Regenerated the documentation with a clean Doxygen warning pass.

### Fixed

- Fixed iterator edge cases for empty `Buffer` storage and aligned container iterator aliases.
- Fixed arithmetic behavior for broadcast-compatible arrays and scalars, including empty-array cases.
- Fixed stale binding and package documentation that referenced the previous Python layering.

### Removed

- Removed legacy Python wrapper layering and split stub files.
- Stopped tracking generated Python extension binaries.

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
