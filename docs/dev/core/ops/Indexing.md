# Indexing

Developer notes for `include/stratax/core/ops/indexing.hpp`.

## Purpose

Computes flat row-major offsets from a `Shape`, matching `Strides`, and an index container.

## Main API

### Offset Calculation
- `offset(shape, strides, index)`

## Invariants

- Offset calculation assumes `strides` was generated for the same logical shape.
- Valid normalized indexes satisfy `0 <= index[i] < shape[i]` for every dimension.
- Returned offsets are flat row-major positions.
- Rank-zero shape with rank-zero index maps to offset `0`.
- Offset arithmetic must not overflow.

## Validation Notes

- `shape.rank()`, `strides.rank()`, and `index.size()` must match through `core::validation::require_rank()`.
- Rank mismatch throws `Exceptions::DimensionError`.
- Negative indexes are normalized through `core::validation::normalize_index()`.
- Any normalized index outside `[0, dim)` throws `Exceptions::IndexError`.
- Zero-sized dimensions reject all indexes in that dimension.
- Offset multiplication and addition use `core::validation::checked_multiply()` and `core::validation::checked_add()`.

## Implementation Notes

- The function is generic over any index container with:
  - `size()`
  - `begin()`
  - `end()`
- Offset calculation uses `sum(normalized_index[i] * strides[i])`.
- Rank-zero shapes with an empty index return offset `0`.
- Tensor multi-index access depends on this helper.

## Time Complexity

- `offset(shape, strides, index)` is `O(r)`, where `r` is rank.
- Rank validation is `O(1)`.

## Future Work

- Consider accepting initializer lists directly.
- Consider project-specific index container types.
