@page ops_slice Slice Ops

# Slice Operations {#dev_ops_slice}

Version: v0.2.0

Status: Complete

Header: `include/stratax/indexing/Slicing.hpp`

---

## Overview

The slicing API copies strided selections from `Vector`, `Matrix`, and
`Tensor` containers into new owning containers. It resolves raw
`stratax::core::Slice` bounds against concrete dimension extents, supports
positive and negative steps, clamps bounds, and preserves traversal order.

```cpp
const stratax::Vector<int> source{0, 1, 2, 3, 4, 5};
const auto result = stratax::indexing::slice(
    source,
    stratax::core::Slice{1, 6, 2});

// result contains {1, 3, 5}.
```

---

## Responsibilities

The slicing module is responsible for:

- Resolving signed raw bounds against each source dimension
- Translating negative bounds relative to the dimension extent
- Clamping bounds for positive and negative traversal
- Computing output shapes
- Copying selected values in row-major result order
- Checking Tensor result-offset arithmetic

It does not provide non-owning views, omitted-bound syntax, broadcasting, or
advanced gather indexing.

---

## Type Aliases

```cpp
using size_type = std::size_t;
using difference_type = std::ptrdiff_t;
```

`size_type` represents extents, output counts, and flat offsets.
`difference_type` represents normalized signed positions and steps.

---

## Slice Normalization

The internal `detail::normalize_slice` helper converts one raw Slice into:

```cpp
struct ResolvedSlice
{
    difference_type start;
    difference_type step;
    size_type size;
};
```

Normalization is O(1):

- An extent larger than `difference_type::max()` is rejected.
- Negative bounds are translated relative to the extent.
- Positive-step bounds are clamped to `[0, extent]`.
- Negative-step bounds are clamped to `[-1, extent - 1]`.
- For a negative step, a stop value of `-1` remains the reverse-range
  sentinel instead of being translated.
- Directionally empty ranges resolve to size zero.

For example, `Slice{-1, -1, -1}` selects a complete dimension in reverse
order.

---

## Vector Slicing

```cpp
template<typename T>
stratax::container::Vector<T> slice(
    const stratax::container::Vector<T>& vec,
    const stratax::core::Slice& slice);
```

The slice is resolved against `vec.size()`. Selected values are copied into a
new rank-one Vector in traversal order.

```cpp
const Vector<int> values{0, 1, 2, 3, 4};

auto odds = slice(values, Slice{1, 5, 2});       // {1, 3}
auto reversed = slice(values, Slice{-1, -1, -1}); // {4, 3, 2, 1, 0}
```

Throws:

- `Exceptions::IndexError("Vector slice out of bounds.")` if the source
  extent cannot be represented by `difference_type`
- `std::bad_alloc` if result allocation fails

Complexity: O(k), where `k` is the result size.

---

## Matrix Slicing

```cpp
template<typename T>
stratax::container::Matrix<T> slice(
    const stratax::container::Matrix<T>& mat,
    const stratax::core::Slice& rows,
    const stratax::core::Slice& cols);
```

Row and column ranges are resolved independently. The returned Matrix has shape
`{resolved_rows.size, resolved_cols.size}` and owns a row-major copy of the
selected rectangle.

```cpp
const Matrix<int> matrix{
    { 0,  1,  2,  3},
    { 4,  5,  6,  7},
    { 8,  9, 10, 11}
};

auto result = slice(matrix, Slice{0, 3, 2}, Slice{1, 4, 2});
// Shape{2, 2}, values {1, 3, 9, 11}.
```

Throws:

- `Exceptions::IndexError("Matrix row slice out of bounds.")` if the row
  extent cannot be represented
- `Exceptions::IndexError("Matrix column slice out of bounds.")` if the
  column extent cannot be represented
- `std::bad_alloc` if result allocation fails

Complexity: O(result.rows() * result.cols()).

---

## Variadic Tensor Slicing

```cpp
template<typename T, typename... Slices>
stratax::container::Tensor<T> slice(
    const stratax::container::Tensor<T>& tensor,
    Slices... slices);
```

Every variadic argument must be exactly `core::Slice`, enforced with a
compile-time assertion. Exactly one Slice must be supplied per tensor
dimension.

Each dimension is normalized independently. Its selected count becomes the
corresponding output dimension, and selected values are copied into the result
in row-major order.

```cpp
Tensor<int> tensor(Shape{2, 3, 4});

auto result = slice(
    tensor,
    Slice{0, 2},
    Slice{0, 3, 2},
    Slice{1, 4, 2});

// result.shape() == Shape{2, 2, 2}
```

Throws:

- `Exceptions::IndexError("Tensor slice rank must match tensor rank.")` if
  the number of Slice arguments differs from `tensor.rank()`
- `Exceptions::IndexError("Tensor slice out of bounds.")` if a dimension
  extent cannot be represented by `difference_type`
- `Exceptions::DimensionError("Tensor slice offset overflow.")` if checked
  source-offset arithmetic overflows
- `Exceptions::DimensionError` if output shape or stride arithmetic overflows
- Any allocation or element-copy exception propagated while constructing the
  owning result

Complexity: O(result.size() * tensor.rank()).

An empty selected dimension returns an empty Tensor with the fully resolved
shape. A rank-zero Tensor can be sliced by supplying no Slice arguments.

---

## Vector-based Tensor Slicing

```cpp
template<typename T>
stratax::container::Tensor<T> slice(
    const stratax::container::Tensor<T>& tensor,
    const std::vector<stratax::core::Slice>& slices);
```

This overload performs the same normalization and copy operation but accepts a
runtime-sized vector of Slice objects.

```cpp
const std::vector<Slice> ranges{
    Slice{0, 2},
    Slice{0, 3, 2},
    Slice{1, 4, 2}
};

auto result = slice(tensor, ranges);
```

Throws:

- `Exceptions::DimensionError("Slice rank must match tensor rank.")` if
  `slices.size() != tensor.rank()`
- `Exceptions::IndexError("Tensor slice out of bounds.")` if a dimension
  extent cannot be represented by `difference_type`
- `Exceptions::DimensionError("Tensor slice offset overflow.")` if checked
  source-offset arithmetic overflows
- `Exceptions::DimensionError` if output shape or stride arithmetic overflows
- Any allocation or element-copy exception propagated while constructing the
  owning result

Complexity: O(result.size() * tensor.rank()).

For equivalent ranges, this overload produces the same shape and values as the
variadic overload.

---

## Ownership and Empty Results

Every overload returns an independent owning container. Modifying a result does
not modify the source, and result storage does not alias source storage.

Out-of-range bounds are normally clamped rather than rejected. Directionally
empty ranges and zero-sized source dimensions produce correctly shaped empty
containers without entering the element-copy loops.

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Normalize one Slice | O(1) |
| Vector slicing | O(result.size()) |
| Matrix slicing | O(result.rows() * result.cols()) |
| Tensor slicing | O(result.size() * tensor.rank()) |

All overloads allocate storage proportional to the number of selected
elements, plus output shape/stride metadata.

---

## Design Notes

Slicing currently materializes owning copies. This keeps lifetime and mutation
semantics straightforward but makes even contiguous selections O(n).

The two Tensor overloads intentionally expose different rank-error types in the
current implementation: the variadic overload throws `IndexError`, while the
vector-based overload uses `require_rank` and throws `DimensionError`.

---

## Future Improvements

- Deduplicate the two Tensor copy implementations
- Make normalization arithmetic safe across the full signed range
- Harmonize Tensor rank-error types and messages
- Support omitted bounds and full-range shorthand
- Add non-owning strided views

---

## See Also

- @ref slice
- @ref vector
- @ref matrix
- @ref tensor
