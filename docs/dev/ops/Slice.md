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

## Invariants

- Slicing uses half-open ranges `[start, stop)`.
- Slice results are owning containers, not views.
- Vector slice rank is always `1`.
- Matrix slice rank is always `2`.
- Tensor slice rank matches the input tensor rank.
- Result shape dimensions are the sizes of the supplied slices.
- Slicing preserves source element order within the selected region.

## Validation Notes

- `core::Slice` rejects reversed ranges at construction.
- Vector slice stop must be within `vector.size()` through `core::validation::require_at_most()`.
- Matrix row and column slice stops must be within `rows()` and `cols()` through `core::validation::require_at_most()`.
- Tensor slice count must match tensor rank through `core::validation::require_rank()`.
- Tensor rank mismatch throws `Exceptions::DimensionError`.
- Out-of-bounds slices throw `Exceptions::IndexError`.

## Implementation Notes

- Slices are half-open: `[start, stop)`.
- Vector and matrix slicing copy values into new owning containers.
- Tensor slicing computes result shape from slice sizes.
- Empty result slices return immediately to avoid indexing through zero strides.
- Tensor source offsets are computed from source strides.
- Rank and slice-stop checks should stay routed through `Validation.hpp`.

## Time Complexity

- Vector slicing is `O(k)`, where `k` is the slice length.
- Matrix slicing is `O(rows_selected * cols_selected)`.
- Tensor slicing is `O(result_size * r)`, where `r` is tensor rank.
- Empty result slices are `O(r)` for validation and shape construction.

## Future Work

- Add step support to `core::Slice`.
- Add negative indexing only if signed indexing becomes part of the API.
- Consider slice views to avoid copies.
