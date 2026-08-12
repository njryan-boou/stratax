@page tensor Tensor

# Tensor {#dev_tensor}

Version: v0.2.0

Status: Complete

Header: `include/stratax/containers/Tensor.hpp`

---

## Overview

`Tensor<T>` is an N-dimensional array container for numeric types.

It stores values in contiguous row-major memory and pairs storage with `Shape` and `Strides` metadata for generic multi-dimensional indexing and container interoperability.

---

## Responsibilities

The `Tensor` class is responsible for:

- Owning contiguous N-dimensional element storage
- Exposing shape and stride metadata
- Providing unchecked flat access and checked signed multi-index access
- Providing iterator access over contiguous storage
- Supporting copy/move ownership semantics

The `Tensor` class is **not** responsible for:

- Broadcasting policy decisions
- High-level numerical algorithms
- Flat checked single-index `at(index)` access (provided by `Vector`)
- Specialized rank-1/rank-2 APIs (handled by `Vector`/`Matrix`)

---

## Relationships

```text
Tensor<T>
│
├── shape_   : core::Shape
├── strides_ : core::Strides
└── buffer_  : core::Buffer<T>
```

Depends on:

- Buffer
- Shape
- Strides
- Indexing helpers (`stratax::indexing::offset`, `stratax::indexing::normalize_index`)
- Exceptions (`Exceptions::IndexError`, `Exceptions::DimensionError`)

Used by:

- Generic tensor algorithms
- Conversion paths to/from vector/matrix representations

Related classes:

- Buffer
- Shape
- Strides
- Vector
- Matrix

---

## Internal Data

| Member | Description |
| ------- | ----------- |
| `core::Shape shape_` | N-dimensional shape metadata |
| `core::Strides strides_` | Row-major stride metadata |
| `core::Buffer<T> buffer_` | Contiguous element storage |

---

## Invariants

The following conditions are always true:

- `size()` equals `buffer_.size()`.
- `rank()` equals `shape().rank()`.
- `strides()` is derived from `shape()` using row-major layout.
- `data()` points to contiguous storage when non-empty.
- Checked `at(...)` overloads validate signed multi-index bounds and support negative components.

---

## Public Interface

## Iterator Aliases

```cpp
using iterator = typename core::Buffer<T>::iterator;
using const_iterator = typename core::Buffer<T>::const_iterator;
using reverse_iterator = typename core::Buffer<T>::reverse_iterator;
using const_reverse_iterator = typename core::Buffer<T>::const_reverse_iterator;
```

`Tensor<T>` mirrors `Buffer<T>` iterator aliases for flat contiguous-storage traversal.

---

## Constructors

### Default Constructor

```cpp
Tensor() noexcept;
```

Constructs an empty rank-0 tensor.

Complexity

- O(1)

Throws

- None

---

### Shape Constructor

```cpp
explicit Tensor(const core::Shape& shape);
```

Constructs tensor storage for `shape` with default-initialized elements.

Complexity

- O(n)

Throws

- `std::bad_alloc`
- `Exceptions::DimensionError` on element-count overflow

---

### Shape + Fill Constructor

```cpp
Tensor(const core::Shape& shape, const T& value);
```

Constructs tensor storage for `shape` and fills every element with `value`.

Complexity

- O(n)

Throws

- `std::bad_alloc`
- `Exceptions::DimensionError` on element-count overflow

---

### Copy Constructor

```cpp
Tensor(const Tensor&) = default;
```

Complexity

- O(n)

---

### Move Constructor

```cpp
Tensor(Tensor&&) noexcept = default;
```

Complexity

- O(1)

---

### Destructor

```cpp
~Tensor() = default;
```

Complexity

- O(n)

---

## Assignment Operators

## Copy Assignment

```cpp
Tensor& operator=(const Tensor&) = default;
```

Complexity

- O(n)

---

## Move Assignment

```cpp
Tensor& operator=(Tensor&&) noexcept = default;
```

Complexity

- O(1)

---

## Methods

## size()

```cpp
[[nodiscard]] std::size_t size() const noexcept;
```

Returns number of stored elements.

Complexity

- O(1)

---

## empty()

```cpp
[[nodiscard]] bool empty() const noexcept;
```

Returns whether tensor stores no elements.

Complexity

- O(1)

---

## rank()

```cpp
[[nodiscard]] std::size_t rank() const noexcept;
```

Returns number of dimensions.

Complexity

- O(1)

---

## shape() / strides()

```cpp
const core::Shape& shape() const noexcept;
const core::Strides& strides() const noexcept;
```

Returns tensor metadata.

Complexity

- O(1)

---

## at() multi-index

```cpp
template<typename... Rest>
requires ((std::is_integral_v<Rest>) && ...)
T& at(std::ptrdiff_t first, std::ptrdiff_t second, Rest... rest);

template<typename... Rest>
requires ((std::is_integral_v<Rest>) && ...)
const T& at(std::ptrdiff_t first, std::ptrdiff_t second, Rest... rest) const;
```

Returns element by signed multi-index with rank and bounds validation.

Complexity

- O(r), where r is rank

Throws

- `Exceptions::IndexError` if rank mismatch or any component is out of bounds

---

## front() / back()

```cpp
T& front();
const T& front() const;
T& back();
const T& back() const;
```

Returns first/last flat element.

Preconditions

- Tensor must not be empty.

Complexity

- O(1)

---

## data()

```cpp
[[nodiscard]] T* data() noexcept;
[[nodiscard]] const T* data() const noexcept;
```

Returns pointer to contiguous storage.

Complexity

- O(1)

---

## Iterators

```cpp
[[nodiscard]] iterator begin() noexcept;
[[nodiscard]] const_iterator begin() const noexcept;
[[nodiscard]] const_iterator cbegin() const noexcept;
[[nodiscard]] iterator end() noexcept;
[[nodiscard]] const_iterator end() const noexcept;
[[nodiscard]] const_iterator cend() const noexcept;
[[nodiscard]] reverse_iterator rbegin() noexcept;
[[nodiscard]] const_reverse_iterator rbegin() const noexcept;
[[nodiscard]] const_reverse_iterator crbegin() const noexcept;
[[nodiscard]] reverse_iterator rend() noexcept;
[[nodiscard]] const_reverse_iterator rend() const noexcept;
[[nodiscard]] const_reverse_iterator crend() const noexcept;
```

Provides forward and reverse iteration over flat contiguous storage.

Complexity

- O(1)

---

## fill()

```cpp
void fill(const T& value);
```

Assigns `value` to every element.

Complexity

- O(n)

---

## swap()

```cpp
void swap(Tensor& other) noexcept;
```

Exchanges metadata and storage with another tensor.

Complexity

- O(1)

---

## Operators

## operator() multi-index

```cpp
template<typename... Rest>
T& operator()(std::size_t first, std::size_t second, Rest... rest);

template<typename... Rest>
const T& operator()(std::size_t first, std::size_t second, Rest... rest) const;
```

Returns element by multi-index. Validates rank/bounds through `offset(shape_, strides_, indices)`.

Complexity

- O(r), where r is rank

Throws

- `Exceptions::DimensionError` on rank mismatch
- `Exceptions::IndexError` on out-of-bounds component/offset issues

---

## operator() vector-based multi-index

```cpp
T& operator()(const std::vector<std::size_t>& indices);
const T& operator()(const std::vector<std::size_t>& indices) const;
```

Returns element by multi-index from a vector of indices. Validates rank/bounds through `offset(shape_, strides_, indices)`.

Usage

```cpp
Tensor<int> t(Shape{2, 3, 4});
std::vector<std::size_t> idx{1, 2, 3};
int value = t(idx);  // Same as t(1, 2, 3)
```

Complexity

- O(r), where r is rank

Throws

- `Exceptions::DimensionError` on rank mismatch
- `Exceptions::IndexError` on out-of-bounds component/offset issues

---

## operator[]

```cpp
T& operator[](std::size_t index) noexcept;
const T& operator[](std::size_t index) const noexcept;
```

Unchecked flat indexing.

Complexity

- O(1)

See Also

- Checked signed multi-index access: `at(first, second, ...)`

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Default construction | O(1) |
| Shape/shape+fill construction | O(n) |
| Copy construction | O(n) |
| Move construction | O(1) |
| Copy assignment | O(n) |
| Move assignment | O(1) |
| Destruction | O(n) |
| `size()` / `rank()` / `empty()` | O(1) |
| `shape()` / `strides()` | O(1) |
| Flat indexing (`operator[]`) | O(1) |
| Multi-index (`operator()`, `at`) | O(r) |
| `front()` / `back()` | O(1) |
| Iteration | O(n) |
| `fill()` | O(n) |
| `swap()` | O(1) |

---

## Examples

## Creating Tensors

```cpp
core::Shape shape(2, 3, 4);

stratax::container::Tensor<double> a;
stratax::container::Tensor<double> b(shape);
stratax::container::Tensor<double> c(shape, 1.0);
```

---

## Accessing Elements

```cpp
stratax::container::Tensor<double> t(core::Shape(2, 2, 2), 0.0);

t(1, 0, 1) = 5.0;
auto a = t(1, 0, 1);
auto b = t.at(-1, 0, -1);
```

---

## Iteration

```cpp
for (const auto& value : t)
{
    std::cout << value << '\n';
}
```

---

## Design Notes

`Tensor<T>` keeps shape/stride metadata explicit while storing values in one contiguous `Buffer<T>`. Multi-indexing is implemented via offset computation, which centralizes rank and stride logic.

Unchecked access is available for performance-critical code paths, while checked `at(...)` overloads provide safe, signed indexing.

---

## Future Improvements

- Add non-owning tensor view/slice types
- Add reshape/view utilities preserving storage where valid
- Add optimized kernels for common tensor operations

---

## See Also

- @ref buffer
- @ref shape
- @ref strides
- @ref vector
- @ref matrix
