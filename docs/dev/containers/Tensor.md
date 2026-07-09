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

- Flat `operator()` is unchecked.
- Multi-index `operator()` uses `offset()` and throws for rank or bounds errors.
- Empty tensors are possible through the default constructor.
- Zero-dimension shapes are supported and produce empty tensors.

## Implementation Notes

- `shape_`, `strides_`, and `buffer_` must stay in sync.
- Multi-index access depends on `core::Strides` and `ops/Indexing.hpp`.
- Public indexing should use `operator()`, not `operator[]`.
- `operator[]` is available for flat storage indexing.
- Iteration exposes flat storage order.

## Future Work

- Reintroduce shared validation helpers if `Validation.hpp` returns.
- Add initializer-list tensor construction if desired.
- Add broadcasting-aware access helpers.
