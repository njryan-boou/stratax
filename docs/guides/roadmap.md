@page roadmap "Stratax Roadmap"

# Stratax Roadmap

Developer roadmap for the current Stratax implementation and its next stages.

## Purpose

This roadmap tracks verified library capabilities separately from planned work.
A checked item should have a public implementation and automated coverage. Work
that is exposed but not yet tested or documented remains explicitly incomplete.

## Current Priorities

1. Add dedicated Python broadcasting regression tests.
2. Add build, test, and package smoke-test workflows beyond documentation CI.
3. Resolve public-header and documentation maintenance debt.
4. Complete the core shape-transformation API.
5. Establish foundational linear algebra operations before larger numerical modules.

## Phases

### Phase 1: Core Infrastructure

- [x] Buffer
- [x] Shape
- [x] Strides
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
- [x] Copy-based slicing
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
- [ ] Element-wise ordering comparisons
- [ ] Explicit mixed-value-type promotion policy

### Phase 4: Broadcasting

- [x] Trailing-dimension compatibility validation
- [x] Broadcasted result-shape calculation
- [x] Singleton-dimension index projection
- [x] Array-array operation dispatch
- [x] Array-scalar and scalar-array dispatch
- [x] Arithmetic integration
- [x] C++ unit coverage
- [x] Developer documentation
- [ ] Python-specific broadcasting tests
- [x] Broadcasting behavior documented in the user and Python API guides
- [ ] Optimized contiguous and repeated-block kernels
- [ ] Mixed container or result-type policy

### Phase 5: Shape Operations

- [x] reshape
- [x] flatten
- [ ] ravel
- [ ] squeeze
- [ ] expand_dims
- [ ] transpose
- [ ] swapaxes
- [ ] permute_axes
- [ ] Non-owning views

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
- [x] Negative indexing and copy-based slicing
- [x] Reshape and flatten methods
- [x] Single public typing stub
- [ ] Bitwise operator bindings
- [ ] Python broadcasting regression tests
- [ ] Package import smoke test in CI
- [ ] Wheel smoke tests for supported Python versions

### Phase 10: C++ Public Facade

- [x] Top-level aliases for common array types
- [x] Top-level aliases for common algorithms
- [x] Module-style creation, conversion, reduction, transform, and slicing namespaces
- [x] Broadcasting helpers and grouped namespace
- [x] Public exception aliases
- [ ] Remove duplicated umbrella-header declarations
- [ ] Add dedicated namespaces as linear algebra, random, statistics, FFT, and I/O land

### Phase 11: Testing and CI

- [x] C++ unit tests
- [x] Python unit tests
- [x] Dedicated broadcasting test suite
- [x] Documentation check workflow
- [x] Documentation deployment workflow
- [x] Markdown link checking in CI
- [ ] C++ build-and-test workflow
- [ ] Python build-and-test workflow
- [ ] Windows, Linux, and macOS build matrix
- [ ] Sanitizer jobs
- [ ] Release and package smoke tests

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
- [ ] Refresh architecture documentation to match the current source layout
- [ ] Complete generated C++ API reference coverage
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

- Checked items represent behavior found in the current public headers and tests.
- Broadcasting currently applies to arithmetic; comparison remains exact and shape-sensitive.
- Python containers expose `double` specializations of the C++ containers.
- Slicing and reshape return owning results; non-owning views are not implemented.
- Empty and zero-dimension behavior has targeted coverage in implemented containers and operations.
- Documentation CI exists, but code build/test CI is still pending.

## Maintenance Debt

- [ ] Deduplicate the repeated `stratax` facade block in `include/stratax.h`.
- [x] Reconcile README, user-guide, and Python-reference broadcasting claims with the implementation.
- [ ] Refresh architecture notes that still describe validation and Python bindings as future work.
- [ ] Add Python broadcasting tests before treating cross-language broadcasting as fully verified.
- [ ] Keep CMake, package, and documented version metadata synchronized for each release.

## Implementation Principles

- Preserve container storage and shape invariants before adding optimized paths.
- Keep generic behavior shared across Vector, Matrix, and Tensor.
- Pair each new public feature with focused tests and user-facing documentation.
- Prefer clear ownership semantics before introducing views or lazy expressions.
- Establish portable build/test CI before expanding the supported backend matrix.
