@page roadmap "Stratax Roadmap"

# Stratax Roadmap

Developer roadmap for the current Stratax implementation and its next stages.

## Purpose

This roadmap tracks verified library capabilities separately from planned work.
A checked item should have a public implementation and automated coverage. Work
that is exposed but not yet tested or documented remains explicitly incomplete.

## Current Priorities

1. Complete the 0.4.0rc1 release-artifact rehearsal on Linux, Windows, and macOS.
2. Define reverse-view behavior and extend numerical dtype-pair coverage.
3. Complete shape transformations and establish foundational linear algebra.
4. Keep documentation examples executable as APIs change.

## Phases

### Phase 1: Core Infrastructure

- [x] Buffer
- [x] Shape
- [x] Row-major stride metadata through `Shape::strides()`
- [x] Slice descriptors
- [x] Numeric types and concepts
- [x] Exception hierarchy
- [x] Configuration constants
- [x] Shape, dimension, index, and type validation

### Phase 2: Containers

- [x] Vector
- [x] Matrix
- [x] Tensor
- [x] Contiguous owning storage
- [x] Negative indexing
- [x] Shared positive-step slicing
- [x] Creation helpers
- [x] Conversion helpers

### Phase 3: Generic Operators

- [x] Same-shape arithmetic
- [x] Array-scalar and scalar-array arithmetic
- [x] Broadcasted array arithmetic
- [x] Compound and unary arithmetic
- [x] Equality and inequality comparison
- [x] Index offset calculation
- [x] Slicing operations
- [x] Integral bitwise operators
- [ ] Logical operators
- [x] Element-wise ordering comparisons
- [x] Explicit mixed-value-type promotion policy

### Phase 4: Broadcasting

- [x] Trailing-dimension compatibility validation
- [x] Broadcasted result-shape calculation
- [x] Singleton-dimension index projection
- [x] Array-array operation dispatch
- [x] Array-scalar and scalar-array dispatch
- [x] Arithmetic integration
- [x] C++ unit coverage
- [x] Developer documentation
- [x] Python-specific broadcasting tests
- [x] Broadcasting behavior documented in the user and Python API guides
- [ ] Optimized contiguous and repeated-block kernels
- [x] Mixed container or result-type policy

### Phase 5: Shape Operations

- [x] reshape
- [x] flatten
- [ ] ravel
- [ ] squeeze
- [ ] expand_dims
- [ ] transpose
- [ ] swapaxes
- [ ] permute_axes
- [x] Non-owning views

### Phase 6: Reductions

- [x] sum
- [x] prod
- [x] min and max
- [x] argmin and argmax
- [x] mean
- [x] variance and standard deviation
- [x] Axis reductions
- [x] Negative-axis normalization
- [x] keepdims support
- [x] C++ and Python bindings
- [x] C++ and Python tests

### Phase 7: Linear Algebra

- [ ] Matrix multiplication
- [ ] Dot product
- [ ] Cross product
- [ ] Norms
- [ ] Determinant
- [ ] Inverse
- [ ] Linear solve
- [ ] LU decomposition
- [ ] QR decomposition
- [ ] Cholesky decomposition
- [ ] Singular value decomposition
- [ ] Eigenvalues and eigenvectors

### Phase 8: Input and Output

- [x] Stream printing
- [ ] CSV I/O
- [ ] Binary I/O
- [ ] Serialization policy

### Phase 9: Python Bindings

- [x] `_core` extension module
- [x] `Shape`, `Vector`, `Matrix`, and `Tensor` exports
- [x] Arithmetic and comparison bindings
- [x] Creation helpers
- [x] Conversion helpers
- [x] Reduction helpers
- [x] Negative indexing and owner-retaining positive-step slice views
- [x] Reshape and flatten methods
- [x] Single public typing stub
- [ ] Bitwise operator bindings
- [x] Python broadcasting regression tests
- [x] Package import smoke test in CI
- [x] Installed-wheel test commands for supported Python versions

### Phase 10: C++ Public Header

- [x] Umbrella header includes the implemented core APIs
- [x] Standalone public-header compilation coverage
- [x] Document the current namespace map
- [ ] Decide whether to introduce top-level type/algorithm aliases
- [ ] Add namespaces as linear algebra, random, statistics, FFT, and I/O land

### Phase 11: Testing and CI

- [x] C++ unit tests
- [x] Python unit tests
- [x] Dedicated broadcasting test suite
- [x] Documentation check workflow
- [x] Documentation deployment workflow
- [x] Markdown link checking in CI
- [x] C++ build-and-test workflow
- [x] Python build-and-test workflow
- [x] Windows, Linux, and macOS build matrix
- [x] Sanitizer jobs
- [x] Installed-package tests across the Linux Python version matrix
- [x] Run tests against each built release wheel on its target platform

### Phase 12: Documentation and Examples

- [x] Developer docs for implemented core files
- [x] Developer docs for containers, algorithms, and operations
- [x] Broadcasting developer guide
- [x] User guide
- [x] Python API reference
- [x] C++ and Python examples
- [x] Troubleshooting guide
- [x] Release checklist
- [x] Update README feature/status tables for broadcasting
- [x] Update user and Python guides for broadcasting
- [x] Refresh architecture documentation to match the current source layout
- [x] Complete generated C++ API reference coverage
- [ ] README badges

## Later Numerical Phases

- [ ] Random number generation and distributions
- [ ] Statistics
- [ ] FFT
- [ ] Calculus
- [ ] Optimization
- [ ] Automatic differentiation

## Backend and Storage Exploration

- [ ] SIMD kernels
- [ ] Multithreaded execution policy
- [ ] Sparse arrays
- [ ] GPU backend
- [ ] Memory-mapped storage

## Validation Notes

- Checked implementation items refer to this working tree and its tests; they do
  not establish that every configured remote CI job has passed.
- Broadcasting applies to arithmetic, comparison, bitwise, and binary math.
- C++ scalar promotion and container-result policies are explicit, but scalar
  intermediates still follow native C++ rules.
- Python numeric arrays use double; masks use bool and axis indices use int64.
- Slicing shares storage; reshape and conversions copy. Negative-step views and
  slicing existing Python views remain unsupported.
- C++ compound operations read overlapping operands before mutation. Repeated
  destination offsets receive the last result in logical order.
- The prepared Python package, runtime, and documentation version is 0.4.0rc1;
  CMake uses the numeric base 0.4.0. Preparing these files does not publish the
  candidate. Its changes are recorded in the 0.4.0rc1 changelog entry.
- See @ref test_audit for executed local checks and their limits.

## Maintenance Work

- [x] Synchronize the prepared candidate's package, module, documentation, and CMake base versions.
- [ ] Complete the 0.4.0rc1 release workflow rehearsal and inspect its artifacts.
- [x] Add deterministic adversarial coverage for aliasing and numerical edge cases.
- [x] Automate independent header, documentation, branch coverage, and release-artifact checks.
- [x] Establish reproducible performance baselines.
- [ ] Keep generated API contracts, guides, and executable examples consistent.

## Implementation Principles

- Preserve container storage and shape invariants before adding optimized paths.
- Keep generic behavior shared across Vector, Matrix, and Tensor.
- Pair each new public feature with focused tests and user-facing documentation.
- Preserve explicit ownership and lifetime contracts when extending views or adding lazy expressions.
- Validate new backends through the existing portable build/test CI.
