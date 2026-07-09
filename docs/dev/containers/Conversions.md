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

## Validation Notes

- `to_vector` requires rank 1 input.
- `to_matrix` requires rank 2 input.
- `to_tensor` accepts any supported array rank.
- Invalid rank conversions throw `Exceptions::DimensionError`.
- `astype` requires both source and destination element types to satisfy `Numeric`.
- Cast behavior follows `static_cast`.

## Implementation Notes

- Conversion helpers preserve shape and copy linear buffer values with `operator[]`.
- `to_vector` and `to_matrix` depend on destination constructors from `Shape`.
- `astype` supports `Vector`, `Matrix`, and `Tensor`.
- Returned containers own their own buffers.

## Future Work

- Move conversion helpers into a Stratax namespace.
- Add explicit narrowing or lossy-cast policy if needed.
