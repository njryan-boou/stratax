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

## Invariants

- `rank()` always equals the number of stored stride values.
- Strides generated from a shape have the same rank as that shape.
- For non-empty shapes, the last stride is `1`.
- Strides are row-major.
- Empty shapes produce empty strides.
- Public iteration exposes read-only stride values.

## Validation Notes

- `at()` checks bounds through `core::validation::require_index()`.
- `operator()` is unchecked.
- Empty shapes produce empty strides.
- Stride multiplication uses `core::validation::checked_multiply()` and throws `Exceptions::DimensionError`.

## Implementation Notes

- For shape `(2, 3, 4)`, strides are `(12, 4, 1)`.
- Tensor multi-index access depends on these values.
- Public stride indexing should use `operator()`.
- Iteration exposes read-only stride values.
- Keep stride overflow and bounds checks routed through `Validation.hpp`.
- `operator<<` is inline because this file is header-only.

## Time Complexity

- Default construction, `size()`, `rank()`, `empty()`, element access, `front()`, `back()`, `data()`, iterator access, and `swap()` are `O(1)`.
- Construction from `Shape` is `O(r)`, where `r` is rank.
- Equality, inequality, and stream output are `O(r)`.

## Future Work

- Add column-major or layout policy support.
- Consider explicit layout metadata if multiple memory orders are added.
