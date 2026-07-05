# Types

Developer notes for `include/stratax/core/Types.hpp`.

## Purpose

Defines stable dtype aliases used across Stratax.

## Main API

### Boolean
- `bool_`

### Integers
- `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`

### Floating Point
- `float32`
- `float64`
- `float128`

### Complex
- `complex64`
- `complex128`
- `complex256`

### Indexing
- `index_t`
- `ssize_t`

## Validation Notes

- These are aliases only.
- Acceptance as a container dtype is controlled by `Concepts.hpp`.
- `int8` and `uint8` may be excluded by numeric concepts because they alias char-like types.

## Implementation Notes

- Keep aliases mapped directly to standard C++ types.
- Avoid hiding ownership or storage policy in this file.

## Future Work

- Add dtype traits.
- Add dtype promotion rules.
- Add dtype name/string helpers.
- Revisit `float128` portability.
