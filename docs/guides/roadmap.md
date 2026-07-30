@page roadmap "Stratax Roadmap"

# Stratax Roadmap

Developer notes for planned Stratax features.

## Purpose

Tracks the expected build-out of the library from core infrastructure through containers, generic operations, numerical features, bindings, and documentation.

## Phases

### Phase 1: Core Infrastructure

- [x] Buffer
- [x] Shape
- [x] Strides
- [x] Slice
- [x] Types
- [x] Concepts
- [x] Exceptions
- [x] Config
- [x] Validation

### Phase 2: Containers

- [x] Vector
- [x] Matrix
- [x] Tensor
- [x] Creation helpers
- [x] Conversion helpers

### Phase 3: Generic Operators

- [x] Arithmetic
- [x] Comparison
- [x] Indexing offsets
- [x] Slicing
- [ ] Logical operators
- [x] Bitwise operators
- [ ] Broadcasting

### Phase 4: Shape Operations

- [x] reshape
- [x] flatten
- [ ] ravel
- [ ] squeeze
- [ ] expand_dims
- [ ] transpose
- [ ] swapaxes
- [ ] permute_axes

### Phase 5: Linear Algebra

- [ ] Matrix multiplication
- [ ] Dot product
- [ ] Cross product
- [ ] Determinant
- [ ] Inverse
- [ ] Solve
- [ ] LU
- [ ] QR
- [ ] Cholesky
- [ ] SVD
- [ ] Eigenvalues
- [ ] Eigenvectors
- [ ] Norms

### Phase 6: Input and Output

- [x] Stream printing
- [ ] CSV I/O
- [ ] Binary I/O
- [ ] Serialization policy

### Phase 7: Python Bindings

- [x] `_core` extension module
- [x] Top-level `Shape` export
- [x] Top-level `Vector` export
- [x] Top-level `Matrix` export
- [x] Top-level `Tensor` export
- [x] Creation helpers
- [x] Conversion helpers
- [x] Reduction helpers
- [x] Static typing stubs
- [ ] Bitwise operator bindings
- [x] Single public Python stub
- [ ] Package import smoke test in CI

### Phase 7.5: C++ Public Facade

- [x] Top-level public aliases for common array types
- [x] Top-level public aliases for common algorithms
- [x] Module-style aliases for creation, conversions, reductions, transforms, slicing, arrays, and errors
- [ ] Dedicated future namespaces for linear algebra, random, statistics, FFT, and I/O

### Phase 8: Testing and CI

- [x] C++ unit tests
- [x] Python unit tests
- [x] Documentation deploy workflow
- [x] Documentation check workflow
- [ ] Cross-platform build matrix
- [ ] Release/package smoke test
- [x] Markdown link check in CI

### Phase 9: Documentation and Examples

- [x] Developer docs for implemented core files
- [x] Developer docs for implemented containers
- [x] Developer docs for implemented ops
- [x] Developer docs for printing
- [x] User-facing guide
- [x] Python API reference
- [x] Example refresh
- [x] Troubleshooting guide
- [x] Release checklist
- [x] Documentation CI check
- [x] Resolve Doxygen warnings
- [ ] Complete generated C++ API reference coverage
- [ ] README badges

### Later Phases

- [ ] Calculus
- [ ] Statistics
- [ ] Random number support
- [ ] Optimization

## Validation Notes

- The roadmap describes intended scope, not guaranteed current behavior.
- Checked items should have tests before they are treated as complete.
- Validation is implemented at the core level; broader validation enhancements may still be pending.
- Placeholder headers are intentionally not marked complete until they expose real behavior.
- Empty and zero-dimension shape behavior is tested for implemented containers and ops.

## Implementation Notes

- Complete core and container correctness before expanding into larger numerical features.
- Generic operators should work across all containers before specialized algorithms are added.
- Shape operations should preserve storage invariants and document ownership behavior where it affects users.
- Optimization work should come after correctness tests and baseline behavior are stable.
- Current slicing and reshape APIs return owning results; view types are future work.

## Future Work

- Broadcasting
- Views and slices
- Type promotion rules
- Sparse arrays
- GPU backend
- Automatic differentiation
- FFT
- Symbolic mathematics
