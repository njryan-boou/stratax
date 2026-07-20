# Slice Ops {#dev_ops_slice}

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/ops/Slice.hpp`

---

## Overview

`ops/Slice.hpp` implements slicing for vector, matrix, and tensor containers using `stratax::core::Slice` ranges.

All overloads return owning containers that copy selected elements from the source.

---

## Responsibilities

The slice ops module is responsible for:

- Normalizing signed slice ranges against container extents
- Applying strided selection for 1D, 2D, and ND containers
- Building result shapes and copying selected elements

The slice ops module is not responsible for:

- Returning non-owning views
- Defining `Slice` semantics itself
- Broadcasting or advanced gather indexing

---

## Relationships

```text
slice(...) overloads
├── detail::normalize_slice(...)
├── core::Slice metadata
├── validation::require_rank(...) for tensor rank check
└── validation::checked_* for tensor offset arithmetic
```

Depends on:

- `include/stratax/core/Slice.hpp`
- `include/stratax/core/containers/Vector.hpp`
- `include/stratax/core/containers/Matrix.hpp`
- `include/stratax/core/containers/Tensor.hpp`
- `include/stratax/core/validation/Validation.hpp`

---

## Invariants

The following conditions are always true:

- Output containers are owning copies.
- Slice normalization is half-open and step-aware for both step directions.
- Out-of-range start/stop values are clamped, not rejected.
- Tensor slicing requires slice count equal to tensor rank.
- Empty output shapes return early without element copy loops.

---

## Public Interface

### Vector slicing

```cpp
template<typename T>
stratax::container::Vector<T>
slice(const stratax::container::Vector<T>& vec, const stratax::core::Slice& slice);
```

Behavior

- Normalizes and clamps range against `vec.size()`
- Copies selected elements into a new vector

Throws

- `Exceptions::IndexError` if `vec.size()` exceeds `std::ptrdiff_t` representable range

Complexity

- O(k), where `k` is output size

### Matrix slicing

```cpp
template<typename T>
stratax::container::Matrix<T>
slice(
    const stratax::container::Matrix<T>& mat,
    const stratax::core::Slice& rows,
    const stratax::core::Slice& cols);
```

Behavior

- Normalizes/clamps row and column slices independently
- Copies rectangular strided region into a new matrix

Throws

- `Exceptions::IndexError` if row/column extent exceeds `std::ptrdiff_t` range

Complexity

- O(r_out * c_out)

### Tensor slicing

```cpp
template<typename T, typename... Slices>
stratax::container::Tensor<T>
slice(const stratax::container::Tensor<T>& tensor, Slices... slices);
```

Behavior

- Requires all variadic arguments to be `stratax::core::Slice`
- Requires number of slices to match tensor rank
- Computes output shape from resolved slice sizes
- Copies result in flat order using source/result strides

Throws

- `Exceptions::DimensionError` on rank mismatch
- `Exceptions::IndexError` when an extent exceeds `std::ptrdiff_t` range
- `Exceptions::DimensionError` on offset overflow from checked arithmetic

Complexity

- O(n_out * rank)

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| `normalize_slice` | O(1) |
| Vector `slice` | O(k) |
| Matrix `slice` | O(r_out * c_out) |
| Tensor `slice` | O(n_out * rank) |

---

## Examples

```cpp
const auto v2 = slice(v, stratax::core::Slice(0, 10, 2));
const auto m2 = slice(m, stratax::core::Slice(1, -1, 1), stratax::core::Slice(0, 5, 2));
const auto t2 = slice(t, stratax::core::Slice(0, 2), stratax::core::Slice(0, 3), stratax::core::Slice(0, 4));
```

---

## Design Notes

Normalization is permissive by design: start/stop values are clamped to legal bounds, which keeps slicing behavior predictable and pythonic.

Tensor slicing currently copies into new storage rather than producing a view to preserve ownership simplicity.

---

## Future Improvements

- Add non-owning slice/view support
- Add richer slice composition helpers
- Evaluate vectorized copy kernels for dense slices

---

## See Also

- `include/stratax/core/Slice.hpp`
- `include/stratax/core/containers/Tensor.hpp`
