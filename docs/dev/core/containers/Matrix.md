# Matrix

Developer notes for `include/stratax/core/containers/Matrix.hpp`.

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

## Invariants

- A matrix always has rank `2`.
- `rows() == shape()(0)` and `cols() == shape()(1)`.
- `size() == rows() * cols() == shape().elements() == buffer_.size()`.
- Storage is contiguous row-major order.
- `strides()` matches the matrix shape.
- Initializer-list matrices are rectangular.
- `operator[]` indexes the same flat storage used by `operator()(row, col)`.

## Validation Notes

- `operator()(row, col)` checks row and column bounds with `core::validation::require_index()`.
- Shape construction requires rank 2 through `core::validation::require_rank()`.
- Row/column construction checks size overflow with `core::validation::checked_multiply()`.
- Initializer-list construction rejects ragged rows.
- Empty matrices and zero-column matrices are supported.

## Implementation Notes

- Storage is row-major: `row * cols() + col`.
- `shape_`, `strides_`, and `buffer_` should stay in sync.
- Arithmetic code expects construction from `Shape`.
- `operator[]` is available for flat storage indexing.
- Rank, bounds, and size overflow checks should use `Validation.hpp`.

## Time Complexity

- Metadata access, element access, `front()`, `back()`, `data()`, iterator access, and `swap()` are `O(1)`.
- Row/column, filled, shape, and initializer-list constructors are `O(n)`.
- Initializer-list construction also checks row widths in `O(rows)`.
- Copy construction, copy assignment, `fill()`, equality through ops, and destruction are `O(n)`.
- Move construction and move assignment are `O(1)`.

## Future Work

- Consider whether `operator[]` should become checked or stay a flat fast path.
- Add matrix-specific linalg integration.
