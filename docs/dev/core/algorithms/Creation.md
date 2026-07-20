# Creation {#dev_creation}

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/algorithms/Creation.hpp`

---

## Overview

`Creation.hpp` provides convenience constructors for common tensor initialization patterns.

All helpers return owning `Tensor<T>` values and require `T` to satisfy `Numeric`.

---

## Responsibilities

The creation module is responsible for:

- Creating tensors filled with zeros, ones, or a constant value
- Creating square identity tensors
- Preserving requested output shape semantics

The creation module is not responsible for:

- Returning specialized `Vector` or `Matrix` containers
- Lazy/view-based construction
- Randomized initialization policies

---

## Relationships

```text
creation::zeros / ones / full
└── Tensor(shape, value)

creation::identity
├── zeros<T>({size, size})
└── diagonal assignment I(i, i) = T{1}
```

Depends on:

- `include/stratax/core/Concepts.hpp`
- `include/stratax/core/containers/Tensor.hpp`
- `include/stratax/core/containers/Shape.hpp`

---

## Invariants

The following conditions are always true:

- All functions return owning `stratax::container::Tensor<T>`.
- `zeros`, `ones`, and `full` preserve the input shape exactly.
- `identity(size)` returns a rank-2 square tensor with shape `(size, size)`.
- Identity diagonal entries are `T{1}` and off-diagonal entries are `T{}`.

---

## Public Interface

### zeros

```cpp
template<typename T>
requires Numeric<T>
stratax::container::Tensor<T> zeros(const stratax::core::Shape& shape);
```

Creates a tensor filled with `T{}`.

### ones

```cpp
template<typename T>
requires Numeric<T>
stratax::container::Tensor<T> ones(const stratax::core::Shape& shape);
```

Creates a tensor filled with `T{1}`.

### full

```cpp
template<typename T>
requires Numeric<T>
stratax::container::Tensor<T> full(const stratax::core::Shape& shape, const T& value);
```

Creates a tensor filled with `value`.

### identity

```cpp
template<typename T>
requires Numeric<T>
stratax::container::Tensor<T> identity(std::size_t size);
```

Creates a square identity tensor.

Throws

- Propagates constructor/allocation exceptions from `Tensor`

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| `zeros` / `ones` / `full` | O(n + r) |
| `identity(size)` | O(size^2) |

`n` is element count and `r` is rank.

---

## Examples

```cpp
const auto z = creation::zeros<double>(stratax::core::Shape(2, 3));
const auto o = creation::ones<float>(stratax::core::Shape(4));
const auto f = creation::full<int>(stratax::core::Shape(2, 2, 2), 7);
const auto I = creation::identity<double>(3);
```

---

## Design Notes

Keeping these helpers in a small `creation` namespace provides concise call sites without adding policy complexity to container constructors.

---

## Future Improvements

- Add explicit matrix-returning identity helper if rank-2 specialization is desired
- Add additional builders (`arange`, `linspace`, random initialization) when needed

---

## See Also

- `include/stratax/core/containers/Tensor.hpp`
