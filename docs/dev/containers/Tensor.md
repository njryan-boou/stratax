@page tensor Tensor

# Tensor {#dev_tensor}

Version: v0.2.0

Status: Complete

Header: `include/stratax/containers/Tensor.hpp`

---

## Overview

`stratax::container::Tensor<T>` is an arbitrary-rank owning array for types
that satisfy the `Numeric` concept. It derives from `core::ArrayBase<T>` and
stores elements contiguously using canonical row-major strides.

The default tensor is empty with shape `{0}` and rank one. Constructing from
an explicit `Shape{}` instead creates an empty rank-zero tensor.

```cpp
stratax::container::Tensor<double> tensor(
    stratax::core::Shape{2, 3, 4},
    0.0);

tensor(1, 2, 3) = 7.0; // unchecked multidimensional access
tensor.at(-1, -1, -1); // 7.0; checked negative indices
tensor.at(-1);         // checked flat access inherited from ArrayBase
```

---

## Responsibilities

`Tensor<T>` is responsible for:

- Owning arbitrary-rank contiguous storage through `ArrayBase<T>`
- Preserving explicit shape and row-major stride metadata
- Providing unchecked variadic and vector-based multidimensional access
- Providing checked signed variadic and vector-based multidimensional access
- Retaining the inherited flat container interface
- Supporting constant-time member and argument-dependent `swap`

`Tensor<T>` does not directly implement broadcasting, reshaping, slicing, or
high-level numerical algorithms.

---

## Representation and Invariants

```text
Tensor<T>
└── core::ArrayBase<T>
    ├── core::Buffer<T> buffer_
    ├── core::Shape     shape_
    └── core::Strides   strides_
```

For every normally constructed tensor:

- `size() == shape().elements()`
- `rank() == shape().rank() == strides().rank()`
- `strides()` describes the canonical row-major layout of `shape()`
- Elements occupy one contiguous memory range

Shapes of any rank are accepted, including rank zero and shapes containing zero
dimensions. Element-count and stride multiplication are checked during
construction.

A moved-from tensor remains destructible and assignable, but its previous
contents and layout must not be relied upon.

---

## Type Aliases

`Tensor<T>` republishes the complete container alias set from
`core::ArrayBase<T>`:

```cpp
using value_type = typename core::ArrayBase<T>::value_type;
using size_type = typename core::ArrayBase<T>::size_type;
using difference_type = typename core::ArrayBase<T>::difference_type;
using reference = typename core::ArrayBase<T>::reference;
using const_reference = typename core::ArrayBase<T>::const_reference;
using pointer = typename core::ArrayBase<T>::pointer;
using const_pointer = typename core::ArrayBase<T>::const_pointer;
using iterator = typename core::ArrayBase<T>::iterator;
using const_iterator = typename core::ArrayBase<T>::const_iterator;
using reverse_iterator = typename core::ArrayBase<T>::reverse_iterator;
using const_reverse_iterator = typename core::ArrayBase<T>::const_reverse_iterator;
```

---

## Constructors

### Default Constructor

```cpp
Tensor();
```

Constructs an empty rank-one tensor with shape `{0}`.

Complexity: O(1).

### Shape Constructor

```cpp
explicit Tensor(const core::Shape& shape);
```

Constructs `shape.elements()` value-initialized elements. For arithmetic
types, value initialization produces zero.

Complexity: O(shape.elements() + shape.rank()).

Throws:

- `Exceptions::DimensionError` if the element count or a stride overflows
- `std::bad_alloc` if allocation fails
- Any exception propagated from `value_type` construction

### Shape and Fill Constructor

```cpp
Tensor(const core::Shape& shape, const_reference value);
```

Constructs `shape.elements()` copies of `value`.

Complexity: O(shape.elements() + shape.rank()).

It has the same overflow and allocation failure conditions as the shape
constructor and may propagate exceptions from the `value_type` copy
constructor.

---

## Copy and Move Semantics

The compiler-generated special members use `ArrayBase<T>` semantics:

```cpp
Tensor(const Tensor&) = default;
Tensor(Tensor&&) = default;
Tensor& operator=(const Tensor&) = default;
Tensor& operator=(Tensor&&) = default;
~Tensor() = default;
```

Copying duplicates element storage and metadata. Moving transfers their
ownership. Copy operations are O(n), while move construction is O(1).

---

## Unchecked Multidimensional Access

### Variadic Indices

```cpp
template<typename... Rest>
requires ((std::is_integral_v<Rest>) && ...)
reference operator()(size_type first, Rest... rest);

template<typename... Rest>
requires ((std::is_integral_v<Rest>) && ...)
const_reference operator()(size_type first, Rest... rest) const;
```

The supplied components are converted to `size_type` and combined with the
row-major strides.

Preconditions:

- Exactly `rank()` components are supplied
- Every component is non-negative
- Every component is smaller than its corresponding dimension

No rank or bounds validation is performed. Violating these preconditions can
produce an invalid offset or undefined behavior.

Complexity: O(rank()).

### Vector-based Indices

```cpp
reference operator()(const std::vector<size_type>& indices);
const_reference operator()(const std::vector<size_type>& indices) const;
```

This overload has the same preconditions and unchecked behavior as the
variadic overload. `indices.size()` must equal `rank()`.

```cpp
std::vector<std::size_t> indices{1, 2, 3};
tensor(indices); // equivalent to tensor(1, 2, 3)
```

Complexity: O(rank()).

---

## Checked Multidimensional Access

### Variadic Signed Indices

```cpp
template<typename... Rest>
requires ((std::is_integral_v<Rest>) && ...)
reference at(difference_type first, Rest... rest);

template<typename... Rest>
requires ((std::is_integral_v<Rest>) && ...)
const_reference at(difference_type first, Rest... rest) const;
```

Exactly one component per tensor dimension must be supplied. Each signed
component is normalized independently, and negative values count backward from
the end of the corresponding dimension.

```cpp
tensor.at(-1, -1, -1); // final element of a rank-three tensor
```

Complexity: O(rank()).

Throws `Exceptions::IndexError` with:

- `"Tensor multi-index rank must match tensor rank."` for a rank mismatch
- `"Tensor multi-index component is out of bounds."` for an invalid component

### Vector-based Signed Indices

```cpp
reference at(const std::vector<difference_type>& raw_indices);
const_reference at(const std::vector<difference_type>& raw_indices) const;
```

The vector overload performs the same rank validation, negative-index
normalization, bounds checking, and error reporting as the variadic overload.

```cpp
std::vector<std::ptrdiff_t> indices{-1, 0, -2};
tensor.at(indices);
```

Complexity: O(rank()).

---

## Inherited Flat Interface

Tensor explicitly retains the one-argument checked `ArrayBase<T>::at`
overloads:

```cpp
reference operator[](size_type index) noexcept;
const_reference operator[](size_type index) const noexcept;

reference at(difference_type index);
const_reference at(difference_type index) const;
```

`operator[]` is unchecked flat access. The one-argument `at(index)` checks a
flat index in `[-size(), size())` and supports negative values. It throws
`Exceptions::IndexError` when the flat index is invalid.

The inherited interface also provides:

- `size()`, `empty()`, `rank()`, `shape()`, and `strides()`
- `data()`, `front()`, and `back()`
- Forward, const, and reverse iterators
- `fill()`

`front()` and `back()` throw `Exceptions::IndexError` when the tensor is
empty.

Metadata queries, iterator acquisition, and individual flat element access are
O(1). Traversal and `fill()` are O(size()).

---

## Swap

```cpp
void swap(Tensor& other) noexcept;
friend void swap(Tensor& lhs, Tensor& rhs) noexcept;
```

Both overloads exchange the buffer, shape, and strides in O(1). The non-member
overload supports argument-dependent lookup:

```cpp
using std::swap;
swap(lhs, rhs);
```

Tensors of different ranks and shapes may be swapped.

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Default construction | O(1) |
| Shape or shape-and-fill construction | O(elements + rank) |
| Copy construction or assignment | O(n) |
| Move construction | O(1) |
| Metadata query | O(1) |
| Flat element access | O(1) |
| Variadic or vector multidimensional access | O(rank) |
| Iterator acquisition | O(1) |
| Complete traversal | O(n) |
| `fill()` | O(n) |
| `swap()` | O(1) |

---

## Examples

### Construction

```cpp
const stratax::core::Shape shape{2, 3, 4};

stratax::container::Tensor<double> empty;
stratax::container::Tensor<double> zeros(shape);
stratax::container::Tensor<double> filled(shape, 1.0);
stratax::container::Tensor<double> rank_zero(
    stratax::core::Shape{});
```

### Checked and Unchecked Access

```cpp
stratax::container::Tensor<double> tensor(
    stratax::core::Shape{2, 2, 2},
    0.0);

tensor(1, 0, 1) = 5.0; // unchecked
tensor.at(1, 0, 1);    // 5.0
tensor.at(-1, 0, -1);  // 5.0
tensor.at(-1);         // checked access to the final flat element
```

### Iteration

```cpp
for (double& value : tensor)
{
    value += 1.0;
}
```

---

## Design Notes

Keeping ownership and flat container behavior in `ArrayBase<T>` gives
`Vector`, `Matrix`, and `Tensor` consistent storage and iterator semantics.
`Tensor<T>` adds general row-major multi-index conversion for arbitrary ranks.

Unchecked access deliberately avoids validation for performance-sensitive code.
Use `at(...)` when indices are external, signed, or otherwise untrusted.

---

## See Also

- @ref arraybase
- @ref buffer
- @ref shape
- @ref strides
- @ref vector
- @ref matrix
