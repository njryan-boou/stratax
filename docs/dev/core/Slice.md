# Slice

Developer notes for `include/stratax/core/Slice.hpp`.

## Purpose

Represents a half-open range used by slicing operations.

## Main API

### Constructor
- `Slice(start, stop)`

### Metadata
- `start()`
- `stop()`
- `size()`
- `empty()`

### Comparison
- `operator==`
- `operator!=`

## Validation Notes

- Slices are half-open: `[start, stop)`.
- `start > stop` is invalid and throws `Exceptions::IndexError`.
- Empty slices are valid when `start == stop`.

## Implementation Notes

- `size()` is `stop - start`.
- Constructor validation prevents unsigned underflow in `size()`.
- Slice bounds against a container are checked by ops-level slicing code.

## Future Work

- Add optional step support.
- Consider signed indexes only if negative indexing becomes part of the API.
