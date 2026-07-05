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
- `size()`
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

- Validation is intentionally minimal while `Validation.hpp` is deferred.
- `operator()` is unchecked.
- `at()` currently mirrors direct storage access.
- Empty vectors are possible through the default constructor.

## Implementation Notes

- `shape_` should stay rank 1 for normal vector instances.
- `buffer_` owns contiguous element storage.
- Public indexing should use `operator()`, not `operator[]`.
- Arithmetic code expects construction from `Shape`.

## Future Work

- Reintroduce validation helpers.
- Add checked `at()` behavior.
- Decide whether zero-size vectors should be allowed.
- Add shape/rank invariants when validation returns.
