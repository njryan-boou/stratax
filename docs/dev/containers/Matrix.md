# Matrix

Developer notes for `include/stratax/containers/Matrix.hpp`.

## Purpose

Implements the rank-2 Stratax array container with row-major contiguous storage.

## Main API

### Constructors
- Default constructor
- `Matrix(core::Shape)`
- `Matrix(rows, cols)`
- `Matrix(rows, cols, value)`
- `Matrix({{...}, {...}})`

### Metadata
- `rows()`
- `cols()`
- `rank()`
- `shape()`
- `strides()`
- `size()`
- `empty()`

### Element Access
- `operator()(row, col)`
- `at(row, col)`
- `front()`
- `back()`
- `data()`

### Iteration
- `begin()` / `end()`
- `cbegin()` / `cend()`
- `rbegin()` / `rend()`
- `crbegin()` / `crend()`

## Validation Notes

- `operator()(row, col)` checks row and column bounds.
- Shape construction requires rank 2.
- Row/column construction checks size overflow.
- Initializer-list construction rejects ragged rows.
- Empty matrices and zero-column matrices are supported.

## Implementation Notes

- Storage is row-major: `row * cols() + col`.
- `shape_`, `strides_`, and `buffer_` should stay in sync.
- Arithmetic code expects construction from `Shape`.
- `operator[]` is available for flat storage indexing.

## Future Work

- Reintroduce shared validation helpers if `Validation.hpp` returns.
- Consider whether `operator[]` should become checked or stay a flat fast path.
- Add matrix-specific linalg integration.
