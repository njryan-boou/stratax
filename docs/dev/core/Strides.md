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
- Stride multiplication checks overflow and throws `Exceptions::DimensionError`.

## Implementation Notes

- For shape `(2, 3, 4)`, strides are `(12, 4, 1)`.
- Tensor multi-index access depends on these values.
- Public stride indexing should use `operator()`.
- Iteration exposes read-only stride values.
- `operator<<` is inline because this file is header-only.

## Future Work

- Add column-major or layout policy support.
- Add validation once `Validation.hpp` returns.
- Consider explicit layout metadata if multiple memory orders are added.
