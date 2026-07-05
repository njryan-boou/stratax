# Stratax Roadmap

Developer notes for planned Stratax feature work.

## Purpose

Tracks the expected build-out of the library from core infrastructure through containers, generic operations, numerical features, bindings, and documentation.

## Main API

### Phase 1: Core Infrastructure
- [x] Buffer
- [x] Shape
- [x] Strides
- [x] Types
- [x] Concepts
- [x] Exceptions
- [x] Config
- [ ] Validation

### Phase 2: Containers
- [x] Vector
- [x] Matrix
- [x] Tensor

### Phase 3: Generic Operators
- [x] Arithmetic
- [x] Comparison
- [ ] Logical operators
- [ ] Bitwise operators

### Phase 4: Shape Operations
- [ ] reshape
- [ ] flatten
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

### Later Phases
- Calculus
- Statistics
- Random number support
- Input/output
- Python bindings
- Optimization
- User documentation

## Validation Notes

- The roadmap describes intended scope, not guaranteed current behavior.
- Checked items should have tests before they are treated as complete.
- Validation remains unchecked because the validation module is currently deferred.

## Implementation Notes

- Complete core and container correctness before expanding into larger numerical features.
- Generic operators should work across all containers before specialized algorithms are added.
- Shape operations should preserve storage invariants and make copy/view semantics explicit.
- Optimization work should come after correctness tests and baseline behavior are stable.

## Future Work

- Broadcasting
- Views and slices
- Complex number support
- Sparse arrays
- GPU backend
- Automatic differentiation
- FFT
- Symbolic mathematics
