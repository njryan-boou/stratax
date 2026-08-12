@page indexing Indexing

# Indexing {#dev_indexing}

Version: v0.2.0

Status: Complete

Header: `include/stratax/indexing/Indexing.hpp`

---

## Overview

`Indexing.hpp` provides core indexing helpers used by multidimensional containers.

The `stratax::indexing::offset(...)` helper maps a rank-matched index container to a row-major storage position using shape and stride metadata. The `stratax::indexing::normalize_index(...)` helper resolves signed indexing into validated non-negative positions.

---

## Responsibilities

The indexing module is responsible for:

- Validating rank agreement among shape, strides, and index inputs
- Validating each index component against the corresponding dimension
- Computing flat offsets safely with overflow checks

The indexing module is **not** responsible for:

- Building or owning shape/stride metadata
- Broadcasting and advanced indexing policy
- Materializing slices/views

---

## Relationships

```text
stratax::indexing::offset(shape, strides, index)
├── require_rank(...)
├── checked_multiply(...)
├── checked_add(...)
└── Exceptions::IndexError translation

stratax::indexing::normalize_index(index, size)
└── signed-to-valid-index normalization
```

Depends on:

- `include/stratax/core/Shape.hpp`
- `include/stratax/core/Strides.hpp`
- `include/stratax/core/validation/Validation.hpp`
- `include/stratax/exceptions/Exceptions.hpp`

Used by:

- Tensor multi-index access paths
- Any row-major operation requiring flat addressing

---

## Invariants

The following conditions are always true:

- `shape.rank() == strides.rank() == index.size()`.
- Valid index components satisfy `index[i] < shape[i]`.
- Returned value is a flat row-major offset.
- Overflow in multiplication/addition is translated to `Exceptions::IndexError`.

---

## Public Interface

## stratax::indexing::offset(...)

```cpp
template<typename IndexContainer>
std::size_t offset(
    const stratax::core::Shape& shape,
    const stratax::core::Strides& strides,
    const IndexContainer& index);
```

Computes `sum(index[i] * strides[i])` after validation.

Requirements on `IndexContainer`:

- `size()` returning rank-compatible count
- `begin()` / `end()` iterators over index components

Throws

- `Exceptions::DimensionError` on rank mismatch
- `Exceptions::IndexError` for out-of-bounds component
- `Exceptions::IndexError` for offset arithmetic overflow

Complexity

- O(r), where `r` is rank

---

## stratax::indexing::normalize_index(...)

```cpp
std::size_t normalize_index(std::ptrdiff_t index, std::size_t size);
```

Normalizes signed indices where negative values count from the end.

Throws

- `Exceptions::IndexError` when `index` is out of bounds

Complexity

- O(1)

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Rank validation | O(1) |
| Component validation + offset accumulation | O(r) |
| Total `offset(...)` | O(r) |

---

## Examples

```cpp
const stratax::core::Shape shape(2, 3, 4);
const stratax::core::Strides strides(shape);
const std::array<std::size_t, 3> index{1, 0, 2};

const std::size_t flat = stratax::indexing::offset(shape, strides, index);
// row-major offset for [1,0,2]

const std::size_t i0 = stratax::indexing::normalize_index(-1, 4);
// i0 == 3
```

---

## Design Notes

This helper intentionally keeps behavior strict and explicit: rank mismatch remains `DimensionError`, while per-component bounds and overflow are surfaced as `IndexError` for callers operating in indexing contexts.

The API is generic over index containers to avoid forcing a project-specific index type.

---

## Future Improvements

- Add overloads for fixed-size index packs where rank is compile-time known.
- Add optional debug diagnostics exposing failing dimension indices.

---

## See Also

- `include/stratax/containers/Tensor.hpp`
- `include/stratax/core/validation/IndexValidation.hpp`
