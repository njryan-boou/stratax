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

## Invariants

- `rank()` always equals the number of stored dimensions.
- Dimensions are stored in declaration order.
- `elements()` is the product of all dimensions, except empty shape returns `0`.
- Any zero dimension makes `elements()` return `0`.
- Shape equality requires the same rank and the same dimension values in order.

## Validation Notes

- `operator()` checks dimension index bounds through `core::validation::require_index()`.
- `elements()` returns total element count using `core::validation::checked_multiply()`.
- Empty shape has element count `0`.
- Element count multiplication checks overflow and throws `Exceptions::DimensionError`.

## Implementation Notes

- Public shape indexing should use `operator()`.
- `Shape` stores dimensions in `Buffer<std::size_t>`.
- Use `rank()` when comparing number of dimensions.
- Keep overflow and bounds checks routed through `Validation.hpp`.
- `operator<<` is inline because this file is header-only.

## Time Complexity

- Default construction, `rank()`, `empty()`, `operator()(index)`, iterator access, and `swap()` are `O(1)`.
- Initializer-list and buffer construction are `O(r)`, where `r` is rank.
- `elements()`, equality, inequality, and stream output are `O(r)`.

## Future Work

- Decide scalar-shape semantics for empty shapes.
- Add shape construction from iterator ranges.
- Add more shape arithmetic helpers.
