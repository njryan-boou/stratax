# Slice

Developer notes for `include/stratax/ops/Slice.hpp`.

## Purpose

Implements range-based slicing for vectors, matrices, and tensors using `core::Slice`.

## Main API

### Vector Slicing
- `slice(vector, slice)`

### Matrix Slicing
- `slice(matrix, row_slice, col_slice)`

### Tensor Slicing
- `slice(tensor, slice_0, slice_1, ...)`

## Validation Notes

- `core::Slice` rejects reversed ranges at construction.
- Vector slice stop must be within `vector.size()`.
- Matrix row and column slice stops must be within `rows()` and `cols()`.
- Tensor slice count must match tensor rank.
- Tensor rank mismatch throws `Exceptions::DimensionError`.
- Out-of-bounds slices throw `Exceptions::IndexError`.

## Implementation Notes

- Slices are half-open: `[start, stop)`.
- Vector and matrix slicing copy values into new owning containers.
- Tensor slicing computes result shape from slice sizes.
- Empty result slices return immediately to avoid indexing through zero strides.
- Tensor source offsets are computed from source strides.

## Future Work

- Add step support to `core::Slice`.
- Add negative indexing only if signed indexing becomes part of the API.
- Consider slice views to avoid copies.
