# Tensor

Developer notes for `include/stratax/containers/Tensor.hpp`.

## Purpose

Implements the general N-dimensional Stratax array container using shape, strides, and flat contiguous storage.

## Main API

### Constructors
- Default constructor
- `Tensor(core::Shape)`
- `Tensor(core::Shape, value)`

### Metadata
- `rank()`
- `shape()`
- `strides()`
- `size()`
- `empty()`

### Element Access
- `operator()(flat_index)`
- `operator()(i, j, ...)`
- `at(flat_index)`
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
- Flat `operator()` is unchecked.
- Multi-index `operator()` uses `offset()` and throws for rank or bounds errors.
- Empty tensors are possible through the default constructor.

## Implementation Notes

- `shape_`, `strides_`, and `buffer_` must stay in sync.
- Multi-index access depends on `core::Strides` and `ops/Indexing.hpp`.
- Public indexing should use `operator()`, not `operator[]`.
- Iteration exposes flat storage order.

## Future Work

- Reintroduce tensor shape validation.
- Add initializer-list tensor construction if desired.
- Add checked full-index `at(i, j, ...)`.
- Add broadcasting-aware access helpers.
