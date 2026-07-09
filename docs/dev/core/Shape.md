# Shape

Developer notes for `include/stratax/core/Shape.hpp`.

## Purpose

Stores tensor dimensions and exposes rank and total element count metadata.

## Main API

### Constructors
- Default constructor
- `Shape({ ... })`

### Metadata
- `elements()`
- `rank()`
- `empty()`

### Element Access
- `operator()(dimension_index)`

### Iteration and Utilities
- `begin()` / `end()`
- `operator==`
- `operator!=`
- `swap()`
- `operator<<`

## Validation Notes

- `operator()` checks dimension index bounds.
- `elements()` returns total element count.
- Empty shape has element count `0`.
- Element count multiplication checks overflow and throws `Exceptions::DimensionError`.

## Implementation Notes

- Public shape indexing should use `operator()`.
- `Shape` stores dimensions in `Buffer<std::size_t>`.
- Use `rank()` when comparing number of dimensions.
- `operator<<` is inline because this file is header-only.

## Future Work

- Decide scalar-shape semantics for empty shapes.
- Add validation helpers once `Validation.hpp` returns.
- Add shape construction from iterator ranges.
- Add more shape arithmetic helpers.
