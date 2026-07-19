# Conversions

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/algorithms/Conversions.hpp`

---

## Overview

`Conversions.hpp` provides shape conversion and value-type casting helpers for Stratax array containers.

Helpers preserve flat storage order and return new owning containers.

---

## Responsibilities

The conversions module is responsible for:

- Converting array-like containers to vector, matrix, or tensor forms
- Supporting shape compatibility checks for vector/matrix conversion
- Casting element types across vector/matrix/tensor containers

The conversions module is not responsible for:

- View-based conversion
- Implicit numeric safety checks beyond `static_cast`
- Runtime dtype-policy configuration

---

## Relationships

```text
to_vector / to_matrix / to_tensor
└── flat index copy loop

astype<To>(...)
└── per-element static_cast<To>
```

Depends on:

- `include/stratax/core/Concepts.hpp`
- `include/stratax/core/Exceptions.hpp`
- `include/stratax/core/containers/Vector.hpp`
- `include/stratax/core/containers/Matrix.hpp`
- `include/stratax/core/containers/Tensor.hpp`

---

## Invariants

The following conditions are always true:

- Conversions return new owning containers.
- Element order is preserved in flat storage order.
- `to_vector` accepts shapes that are rank-1 or have exactly one non-singleton dimension.
- `to_matrix` accepts shapes that are rank-2 or have exactly two non-singleton dimensions.
- `to_tensor` preserves original shape exactly.
- `astype` preserves shape and element count.

---

## Public Interface

### Shape helpers

```cpp
bool is_vector_shape(const stratax::core::Shape& shape);
bool is_matrix_shape(const stratax::core::Shape& shape);
stratax::core::Shape matrix_shape(const stratax::core::Shape& shape);
```

### Container conversions

```cpp
template<Array A>
stratax::container::Vector<typename A::value_type> to_vector(const A& arr);

template<Array A>
stratax::container::Matrix<typename A::value_type> to_matrix(const A& arr);

template<Array A>
stratax::container::Tensor<typename A::value_type> to_tensor(const A& arr);
```

Throws

- `Exceptions::ShapeError` for unsupported source shape in `to_vector`/`to_matrix`

### Type casting

```cpp
template<typename To, typename From>
requires Numeric<To> && Numeric<From>
stratax::container::Vector<To> astype(const stratax::container::Vector<From>& vec);

template<typename To, typename From>
requires Numeric<To> && Numeric<From>
stratax::container::Matrix<To> astype(const stratax::container::Matrix<From>& mat);

template<typename To, typename From>
requires Numeric<To> && Numeric<From>
stratax::container::Tensor<To> astype(const stratax::container::Tensor<From>& tensor);
```

Behavior

- Uses `static_cast<To>` per element

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| `is_vector_shape` / `is_matrix_shape` | O(r) |
| `matrix_shape` | O(r) |
| `to_vector` / `to_matrix` / `to_tensor` | O(n + r) |
| `astype` overloads | O(n) |

`n` is element count and `r` is rank.

---

## Examples

```cpp
const auto v = to_vector(tensor_like);
const auto m = to_matrix(tensor_like);
const auto t = to_tensor(matrix_like);

const auto as_double = astype<double>(v);
```

---

## Design Notes

Vector/matrix conversion is intentionally permissive for singleton dimensions, which simplifies interoperability with tensor-shaped data that carries redundant axes.

---

## Future Improvements

- Add explicit policy helpers for strict rank-only conversion
- Add optional checked-cast helpers for narrowing conversions

---

## See Also

- `include/stratax/core/algorithms/Reshape.hpp`
- `include/stratax/core/containers/Shape.hpp`
