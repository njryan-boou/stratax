# Validation

Developer notes for `include/stratax/core/Validation.hpp`.

## Purpose

Provides small shared runtime validation helpers used by core metadata, containers, ops, and bindings.

## Main API

### Signed Conversion
- `nonnegative_size(value, message)`
- `nonnegative_shape_dimension(value, message)`
- `nonnegative_index(value, message)`

### Rank and Bounds
- `require_rank(actual, expected, message)`
- `require_rank(object, expected, message)`
- `require_index(index, size, message)`
- `require_at_most(value, upper, message)`

### Arithmetic Safety
- `checked_multiply(lhs, rhs, message)`
- `checked_add(lhs, rhs, message)`

### Shape Matching
- `same_shape(lhs, rhs)`
- `require_same_shape(lhs, rhs, message)`
- `require_equal_size(lhs, rhs, message)`

## Invariants

- Helpers do not mutate their inputs.
- Helpers preserve Stratax exception categories:
  - dimension/rank/size failures use `Exceptions::DimensionError`
  - shape dimension and shape compatibility failures use `Exceptions::ShapeError`
  - bounds and signed index failures use `Exceptions::IndexError`
- Checked arithmetic must never wrap.
- `require_rank(object, ...)` returns the original object reference when validation succeeds.

## Validation Notes

- Use `require_index()` for exclusive upper-bound indexing.
- Use `require_at_most()` for half-open slice stops where `stop == size` is valid.
- Use `checked_multiply()` and `checked_add()` before size or offset arithmetic that could overflow.
- Use `require_same_shape()` for throwing array compatibility checks.
- Use `same_shape()` when the caller needs a boolean comparison.

## Implementation Notes

- This header is intentionally header-only.
- Keep helpers small, generic, and allocation-free.
- Keep Python-specific exception translation in the binding files; core helpers should throw Stratax exceptions.
- Prefer passing caller-specific messages instead of baking container names into generic helpers.

## Time Complexity

- All scalar validation helpers are `O(1)`.
- `same_shape()` and `require_same_shape()` are `O(r)` because shape equality is rank-dependent.
- Space complexity is `O(1)` for every helper.

## Future Work

- Add broadcasting validation once broadcasting semantics are finalized.
- Consider helper overloads for rectangular nested input validation.
- Consider specialized offset helpers if more indexing code needs the same overflow translation.
