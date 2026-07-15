# Slice

Developer notes for `include/stratax/core/Slice.hpp`.

## Purpose

Represents a half-open range used by slicing operations.

## Main API

### Constructor
- `Slice(start, stop, step = 1)`

### Metadata
- `start()`
- `stop()`
- `size()`
- `empty()`

### Comparison
- `operator==`
- `operator!=`

## Invariants

- Slices are half-open ranges: `[start, stop)`.
- Slice step is always non-zero.
- Positive steps produce values while `index < stop()`.
- Negative steps produce values while `index > stop()`.
- `empty()` is true exactly when the normalized interval has no selected indices.

## Validation Notes

- Slices are half-open: `[start, stop)`.
- `step == 0` is invalid and throws `Exceptions::IndexError`.
- Positive and negative directions are both valid.
- Empty slices are valid when the chosen direction selects no indices.

## Implementation Notes

- `size()` computes element count from `(start, stop, step)` for both step directions.
- Signed index storage enables negative indexing semantics at higher layers.
- Slice bounds against a container are checked by ops-level slicing code.

## Time Complexity

- Construction, metadata access, `size()`, `empty()`, equality, and inequality are `O(1)`.

## Future Work

- Add helper constructors for common patterns like full-range slices.
