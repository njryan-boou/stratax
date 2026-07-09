# Buffer

Developer notes for `include/stratax/core/Buffer.hpp`.

## Purpose

Implements the low-level aligned contiguous storage owner used by Stratax containers.

## Main API

### Constructors
- Default constructor
- `Buffer(size)`
- `Buffer(size, value)`
- `Buffer(size, Buffer::uninitialized)`
- `Buffer({ ... })`

### Element Access
- `operator[]`
- `front()`
- `back()`
- `data()`

### Capacity
- `size()`
- `empty()`

### Iteration
- `begin()` / `end()`
- `cbegin()` / `cend()`
- `rbegin()` / `rend()`
- `crbegin()` / `crend()`

## Validation Notes

- `Buffer` does not validate indexes.
- `front()` and `back()` throw `Exceptions::IndexError` when the buffer is empty.
- Allocation size checks guard against `sizeof(T) * size` overflow.
- Higher-level containers own shape and bounds policy.
- `uninitialized` storage is restricted to trivially destructible types.

## Implementation Notes

- Allocation uses explicit alignment.
- Copy construction uses `memcpy` for trivially copyable types.
- Non-trivial elements are constructed and destroyed manually.
- `Buffer` is allowed to keep `operator[]` because it is the raw storage primitive.

## Future Work

- Add allocator customization.
- Add debug-only bounds checks.
- Add move/copy stress tests for non-trivial types.
- Consider exposing alignment as a named constant or config value.
