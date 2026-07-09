# Creation

Developer notes for `include/stratax/containers/Creation.hpp`.

## Purpose

Provides convenience functions for constructing tensors with common initial values.

## Main API

### Filled Tensors
- `creation::zeros<T>(shape)`
- `creation::ones<T>(shape)`
- `creation::full<T>(shape, value)`

### Identity Tensor
- `creation::identity<T>(size)`

## Validation Notes

- Creation helpers require `Numeric<T>`.
- Creation helpers rely on `Tensor` constructors for shape allocation.
- Zero-sized dimensions are allowed and produce empty tensors.
- `identity` creates a rank-2 square tensor and assumes `size` is the desired row and column count.

## Implementation Notes

- Helpers currently live in the global `creation` namespace.
- `zeros`, `ones`, and `full` preserve the requested `Shape`.
- `identity` builds from `zeros<T>({size, size})` and writes `T{1}` on the diagonal.
- These helpers return `Tensor<T>` instances.
- Complex numeric values are supported.

## Future Work

- Move creation helpers into a Stratax namespace.
- Consider `Matrix`-specific identity creation.
- Add validation for unsupported identity sizes if stricter rules are added.
