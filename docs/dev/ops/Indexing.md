# Indexing

Developer notes for `include/stratax/ops/indexing.hpp`.

## Purpose

Computes flat row-major offsets from a `Shape`, matching `Strides`, and an index container.

## Main API

### Offset Calculation
- `offset(shape, strides, index)`

## Validation Notes

- `shape.rank()`, `strides.rank()`, and `index.size()` must match.
- Rank mismatch throws `Exceptions::DimensionError`.
- Any index greater than or equal to its matching dimension throws `Exceptions::IndexError`.
- Zero-sized dimensions reject all indexes in that dimension.
- Offset multiplication and addition are checked for overflow.

## Implementation Notes

- The function is generic over any index container with:
  - `size()`
  - `begin()`
  - `end()`
- Offset calculation uses `sum(index[i] * strides[i])`.
- Rank-zero shapes with an empty index return offset `0`.
- Tensor multi-index access depends on this helper.

## Future Work

- Consider accepting initializer lists directly.
- Consider project-specific index container types.
- Add support for negative indexes only if signed indexing becomes a design goal.
