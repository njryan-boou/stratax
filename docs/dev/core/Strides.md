# Strides

Developer notes for `include/stratax/core/Strides.hpp`.

## Purpose

Stores row-major stride values derived from a `Shape`.

## Main API

### Constructors
- Default constructor
- `Strides(shape)`

### Metadata
- `size()`
- `rank()`
- `empty()`

### Element Access
- `operator()(dimension_index)`
- `at(dimension_index)`
- `front()`
- `back()`
- `data()`

### Iteration and Utilities
- `begin()` / `end()`
- `cbegin()` / `cend()`
- `rbegin()` / `rend()`
- `operator==`
- `operator!=`
- `swap()`
- `operator<<`

## Validation Notes

- `at()` checks bounds.
- `operator()` is unchecked.
- Empty shapes produce empty strides.

## Implementation Notes

- For shape `(2, 3, 4)`, strides are `(12, 4, 1)`.
- Tensor multi-index access depends on these values.
- Public stride indexing should use `operator()`.

## Future Work

- Add column-major or layout policy support.
- Add validation once `Validation.hpp` returns.
- Add overflow checks for large shapes.
- Add tests for empty and rank-1 printing.
