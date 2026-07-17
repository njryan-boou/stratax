# Vector

Developer notes for `include/stratax/core/containers/Vector.hpp`.

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

## Invariants

- A shape-constructed vector always has rank `1`.
- `size() == shape().elements() == buffer_.size()`.
- `strides()` has rank `1` for rank-1 vectors and stride value `1`.
- Elements are stored contiguously in flat order.
- `operator[]` and flat `operator()` address the same underlying element.
- Default-constructed vectors are empty and have rank `0`.

## Validation Notes

- `operator()` is unchecked.
- `at()` checks bounds with `core::validation::require_index()` and throws `Exceptions::IndexError`.
- Shape construction requires rank 1 through `core::validation::require_rank()`.
- Empty vectors are possible through the default constructor.
- Zero-size rank-1 vectors are supported.

## Implementation Notes

- `shape_`, `strides_`, and `buffer_` should stay in sync.
- `buffer_` owns contiguous element storage.
- `operator[]` is available for flat storage indexing.
- Arithmetic code expects construction from `Shape`.
- Rank and bounds checks should use `Validation.hpp`.

## Time Complexity

- Default construction, metadata access, element access, `front()`, `back()`, `data()`, iterator access, and `swap()` are `O(1)`.
- Size, fill, shape, and initializer-list constructors are `O(n)`.
- Copy construction, copy assignment, `fill()`, equality through ops, and destruction are `O(n)`.
- Move construction and move assignment are `O(1)`.

## Future Work

- Consider whether `operator()` should remain unchecked.
- Add vector-specific algorithms.
