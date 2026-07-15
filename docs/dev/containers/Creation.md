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

## Invariants

- Creation helpers always return owning `Tensor<T>` values.
- `zeros`, `ones`, and `full` preserve the requested shape exactly.
- Returned tensor size matches `shape.elements()`.
- `identity(size)` always returns shape `(size, size)`.
- Identity diagonal values are `T{1}` and non-diagonal values are `T{}`.

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

## Time Complexity

- `zeros`, `ones`, and `full` are `O(n + r)`, where `n` is element count and `r` is rank.
- `identity(size)` is `O(size * size)` to allocate and zero-fill, plus `O(size)` to write the diagonal.
- Metadata setup is dominated by tensor construction.

## Future Work

- Consider `Matrix`-specific identity creation.
- Add validation for unsupported identity sizes if stricter rules are added.
