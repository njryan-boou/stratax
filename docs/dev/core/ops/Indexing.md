@page indexing Indexing

# Indexing {#dev_indexing}

Version: v0.2.0

Status: Complete

Header: `include/stratax/indexing/Indexing.hpp`

---

## Overview

`Indexing.hpp` provides core indexing helpers used by multidimensional containers.

The `stratax::indexing::offset(...)` helper maps normalized indices to a
row-major storage position using precomputed stride metadata. The
`stratax::indexing::normalize_index(...)` helper resolves signed indexing into
validated non-negative positions.

---

## Responsibilities

The indexing module is responsible for:

- Computing unchecked flat offsets from stride metadata and normalized indices
- Normalizing checked signed indices against a dimension extent

The indexing module is **not** responsible for:

- Building or owning shape/stride metadata
- Broadcasting and advanced indexing policy
- Materializing slices/views

---

## Relationships

```text
stratax::indexing::offset(strides, indices)
└── sum(indices[i] * strides[i])

stratax::indexing::normalize_index(index, size)
└── signed-to-valid-index normalization
```

Depends on:

- `include/stratax/core/Shape.hpp`
- `include/stratax/indexing/Normalize.hpp`

Used by:

- Tensor multi-index access paths
- Any row-major operation requiring flat addressing

---

## Invariants

Callers of `offset(...)` must maintain these preconditions:

- `strides.rank() == indices.size()`.
- Each index is valid for the logical shape that produced the strides.
- Every multiplication and the final sum fit in `std::size_t`.

---

## Public Interface

## stratax::indexing::offset(...)

```cpp
template<typename IndexContainer>
std::size_t offset(
    const stratax::core::Shape& strides,
    const IndexContainer& indices);
```

Computes `sum(indices[i] * strides[i])` without validation.

Requirements on `IndexContainer`:

- `begin()` / `end()` iterators over normalized index components
- Exactly one index per stride

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
| Offset accumulation | O(r) |
| Total `offset(...)` | O(r) |

---

## Examples

```cpp
const stratax::core::Shape shape{2, 3, 4};
const stratax::core::Shape strides = shape.strides();
const std::array<std::size_t, 3> index{1, 0, 2};

const std::size_t flat = stratax::indexing::offset(strides, index);
// row-major offset for [1,0,2]

const std::size_t i0 = stratax::indexing::normalize_index(-1, 4);
// i0 == 3
```

---

## Design Notes

`offset(...)` is intentionally unchecked because its array-container callers
already own synchronized shape and stride metadata. Checked signed indexing is
handled separately before offset calculation.

The API is generic over index containers to avoid forcing a project-specific index type.

---

## Future Improvements

- Add overloads for fixed-size index packs where rank is compile-time known.
- Add an optional checked overload for external callers.

---

## See Also

- `include/stratax/containers/Tensor.hpp`
- `include/stratax/core/validation/IndexValidation.hpp`
