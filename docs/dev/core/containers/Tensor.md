# Tensor

Developer notes for `include/stratax/core/containers/Tensor.hpp`.

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

## Invariants

- `rank() == shape().rank() == strides().rank()` for shape-constructed tensors.
- `size() == shape().elements() == buffer_.size()`.
- `strides()` is generated from `shape()` and uses row-major layout.
- Flat storage order matches stride-based row-major indexing.
- `operator[]` and flat `operator()(index)` address the same underlying element.
- Any zero dimension produces an empty tensor while preserving the requested rank and shape.
- Default-constructed tensors are empty and have rank `0`.

## Validation Notes

- Flat `operator()` is unchecked.
- Multi-index `operator()` uses `offset()` and throws for rank or bounds errors.
- `at(flat_index)` checks bounds with `core::validation::require_index()`.
- Empty tensors are possible through the default constructor.
- Zero-dimension shapes are supported and produce empty tensors.

## Implementation Notes

- `shape_`, `strides_`, and `buffer_` must stay in sync.
- Multi-index access depends on `core::Strides` and `ops/Indexing.hpp`.
- Public indexing should use `operator()`, not `operator[]`.
- `operator[]` is available for flat storage indexing.
- Iteration exposes flat storage order.
- Flat bounds checks should use `Validation.hpp`.

## Time Complexity

- Default construction, metadata access, flat element access, `front()`, `back()`, `data()`, iterator access, and `swap()` are `O(1)`.
- Construction from `Shape` is `O(n + r)`, where `n` is element count and `r` is rank.
- Multi-index access is `O(r)`.
- Copy construction, copy assignment, `fill()`, equality through ops, and destruction are `O(n)`.
- Move construction and move assignment are `O(1)`.

## Future Work

- Add initializer-list tensor construction if desired.
- Add broadcasting-aware access helpers.
