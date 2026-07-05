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

- Validation is intentionally minimal while `Validation.hpp` is deferred.
- `operator()(row, col)` checks row and column bounds.
- Initializer-list construction rejects ragged rows.
- Empty matrices are possible through the default constructor.

## Implementation Notes

- Storage is row-major: `row * cols() + col`.
- `shape_` should stay rank 2 for normal matrix instances.
- Arithmetic code expects construction from `Shape`.
- Public element indexing should use `operator()`.

## Future Work

- Reintroduce dimension validation.
- Add explicit rank checks for `Matrix(core::Shape)`.
- Decide whether zero-row or zero-column matrices should be allowed.
- Add matrix-specific linalg integration.
