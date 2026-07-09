# Vector

Developer notes for `include/stratax/containers/Vector.hpp`.

## Purpose

Implements the rank-1 Stratax array container backed by `core::Shape` and `core::Buffer<T>`.

## Main API

### Constructors
- Default constructor
- `Vector(core::Shape)`
- `Vector(size)`
- `Vector(size, value)`
- `Vector({ ... })`

### Metadata
- `shape()`
- `strides()`
- `size()`
- `rank()`
- `empty()`

### Element Access
- `operator()(index)`
- `at(index)`
- `front()`
- `back()`
- `data()`

### Iteration
- `begin()` / `end()`
- `cbegin()` / `cend()`
- `rbegin()` / `rend()`
- `crbegin()` / `crend()`

## Validation Notes

- `operator()` is unchecked.
- `at()` checks bounds and throws `Exceptions::IndexError`.
- Shape construction requires rank 1.
- Empty vectors are possible through the default constructor.
- Zero-size rank-1 vectors are supported.

## Implementation Notes

- `shape_`, `strides_`, and `buffer_` should stay in sync.
- `buffer_` owns contiguous element storage.
- `operator[]` is available for flat storage indexing.
- Arithmetic code expects construction from `Shape`.

## Future Work

- Reintroduce shared validation helpers if `Validation.hpp` returns.
- Consider whether `operator()` should remain unchecked.
- Add vector-specific algorithms.
