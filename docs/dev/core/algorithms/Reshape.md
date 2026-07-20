# Reshape {#dev_reshape}

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/algorithms/Reshape.hpp`

---

## Overview

`Reshape.hpp` provides shape-changing copy helpers that preserve flat storage order.

`reshape` returns a tensor with a requested shape, while `flatten` returns a rank-1 vector.

---

## Responsibilities

The reshape module is responsible for:

- Validating element-count compatibility for reshape operations
- Copying values in flat storage order
- Producing owning containers for reshaped/flattened output

The reshape module is not responsible for:

- View-based reshape semantics
- Inferred-dimension syntax (`-1` style placeholders)
- In-place shape mutation

---

## Relationships

```text
reshape(arr, shape)
├── validation::require_equal_size(arr.size(), shape.elements(), ...)
└── flat copy into Tensor

flatten(arr)
└── flat copy into Vector
```

Depends on:

- `include/stratax/core/Concepts.hpp`
- `include/stratax/core/containers/Shape.hpp`
- `include/stratax/core/containers/Tensor.hpp`
- `include/stratax/core/containers/Vector.hpp`
- `include/stratax/core/validation/Validation.hpp`

---

## Invariants

The following conditions are always true:

- Reshape and flatten preserve flat element order.
- `reshape` returns `Tensor<value_type>`.
- `flatten` returns `Vector<value_type>`.
- Output size equals source size.
- Both operations return owning containers.

---

## Public Interface

### reshape

```cpp
template<Array A>
stratax::container::Tensor<typename A::value_type>
reshape(const A& arr, const stratax::core::Shape& shape);
```

Throws

- `Exceptions::ShapeError` when target shape element count differs from source size
- Possible propagated shape/overflow exceptions from `shape.elements()`

Complexity

- O(n + r)

### flatten

```cpp
template<Array A>
stratax::container::Vector<typename A::value_type>
flatten(const A& arr);
```

Complexity

- O(n)

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| `reshape` | O(n + r) |
| `flatten` | O(n) |

`n` is element count and `r` is rank.

---

## Examples

```cpp
const auto t = reshape(vec, stratax::core::Shape(2, 3));
const auto v = flatten(t);
```

---

## Design Notes

The module keeps behavior explicit and safe: reshape validates total size first and always performs a full copy.

---

## Future Improvements

- Add non-owning view reshape when view types exist
- Add optional strict reshaping helpers with additional layout checks

---

## See Also

- `include/stratax/core/algorithms/Conversions.hpp`
- `include/stratax/core/containers/Tensor.hpp`
