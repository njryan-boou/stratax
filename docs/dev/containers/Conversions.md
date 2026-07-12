# Conversions

Developer notes for `include/stratax/containers/Conversions.hpp`.

## Purpose

Provides helpers for converting between Stratax container shapes and casting container element types.

## Main API

### Shape Conversions
- `to_vector(array)`
- `to_matrix(array)`
- `to_tensor(array)`

### Type Casting
- `astype<To>(vector)`
- `astype<To>(matrix)`
- `astype<To>(tensor)`

## Invariants

- Conversion helpers return new owning containers.
- Shape-preserving conversions keep element order in flat storage order.
- `to_vector` only produces rank-1 vectors.
- `to_matrix` only produces rank-2 matrices.
- `to_tensor` preserves the input shape exactly.
- `astype` preserves container shape and element count.

## Validation Notes

- `to_vector` requires rank 1 input through `core::validation::require_rank()`.
- `to_matrix` requires rank 2 input through `core::validation::require_rank()`.
- `to_tensor` accepts any supported array rank.
- Invalid rank conversions throw `Exceptions::DimensionError`.
- `astype` requires both source and destination element types to satisfy `Numeric`.
- Cast behavior follows `static_cast`.

## Implementation Notes

- Conversion helpers preserve shape and copy linear buffer values with `operator[]`.
- `to_vector` and `to_matrix` depend on destination constructors from `Shape`.
- `astype` supports `Vector`, `Matrix`, and `Tensor`.
- Returned containers own their own buffers.
- Rank validation should stay routed through `Validation.hpp`.

## Time Complexity

- `to_vector`, `to_matrix`, and `to_tensor` are `O(n + r)`.
- `astype` is `O(n + r)`.
- Rank checks are `O(1)`.

## Future Work

- Move conversion helpers into a Stratax namespace.
- Add explicit narrowing or lossy-cast policy if needed.
