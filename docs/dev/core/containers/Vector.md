# Vector\\<T\\>

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/containers/Vector.hpp`

---

## Overview

`Vector<T>` is Stratax's rank-1 owning array container for numeric types.

It stores values in contiguous memory and pairs that storage with `Shape` and `Strides` metadata to participate in the common container model used across Stratax arrays.

---

## Responsibilities

The `Vector` class is responsible for:

- Owning contiguous rank-1 element storage
- Exposing shape and stride metadata
- Providing checked and unchecked element access
- Providing iterator access over contiguous storage
- Supporting copy/move ownership semantics

The `Vector` class is **not** responsible for:

- Rank-2 or rank-N multidimensional storage semantics
- Broadcasting logic
- High-level numerical algorithms
- Shape transformation outside the rank-1 contract

---

## Relationships

```text
Vector<T>
│
├── shape_   : core::Shape
├── strides_ : core::Strides
└── buffer_  : core::Buffer<T>
```

Depends on:

- Buffer
- Shape
- Strides
- Validation helpers (`validation::normalize_index`, `validation::require_rank`)
- Exceptions (`Exceptions::IndexError`, `Exceptions::DimensionError`)

Used by:

- Matrix/Tensor conversion paths
- Generic Stratax container utilities

Related classes:

- Shape
- Strides
- Buffer
- Matrix
- Tensor

---

## Internal Data

| Member | Description |
| ------- | ----------- |
| `core::Shape shape_` | Shape metadata (typically rank 1, rank 0 for default state) |
| `core::Strides strides_` | Stride metadata for the vector layout |
| `core::Buffer<T> buffer_` | Contiguous element storage |

---

## Invariants

The following conditions are always true:

- `size()` equals `shape().elements()`.
- For constructed rank-1 vectors, `rank() == 1` and `strides().rank() == 1`.
- `data()` is contiguous and compatible with pointer iteration.
- `at(index)` performs bounds validation and supports negative indexing.
- `operator[]` and `operator()` are unchecked.

---

## Public Interface

## Constructors

### Default Constructor

```cpp
Vector() noexcept;
```

Constructs an empty default vector state (rank 0).

Complexity

- O(1)

Throws

- None

---

### Size Constructor

```cpp
explicit Vector(std::size_t size);
```

Constructs a rank-1 vector of `size` default-initialized elements.

Complexity

- O(n)

Throws

- `std::bad_alloc`

---

### Shape Constructor

```cpp
explicit Vector(const core::Shape& shape);
```

Constructs from a validated rank-1 shape.

Complexity

- O(n)

Throws

- `Exceptions::DimensionError` if `shape` is not rank 1
- `std::bad_alloc`

---

### Fill Constructor

```cpp
Vector(std::size_t size, const T& value);
```

Constructs a rank-1 vector of `size` elements, all initialized to `value`.

Complexity

- O(n)

Throws

- `std::bad_alloc`

---

### Initializer List Constructor

```cpp
Vector(std::initializer_list<T> list);
```

Constructs a rank-1 vector from initializer-list values.

Complexity

- O(n)

Throws

- `std::bad_alloc`

---

### Copy Constructor

```cpp
Vector(const Vector&) = default;
```

Complexity

- O(n)

---

### Move Constructor

```cpp
Vector(Vector&&) noexcept = default;
```

Complexity

- O(1)

---

### Destructor

```cpp
~Vector() = default;
```

Complexity

- O(n)

---

## Assignment Operators

## Copy Assignment

```cpp
Vector& operator=(const Vector&) = default;
```

Complexity

- O(n)

---

## Move Assignment

```cpp
Vector& operator=(Vector&&) noexcept = default;
```

Complexity

- O(1)

---

## Methods

## size()

```cpp
[[nodiscard]] std::size_t size() const noexcept;
```

Returns total number of elements.

Complexity

- O(1)

---

## rank()

```cpp
[[nodiscard]] std::size_t rank() const noexcept;
```

Returns the shape rank.

Complexity

- O(1)

---

## empty()

```cpp
[[nodiscard]] bool empty() const noexcept;
```

Returns whether the vector contains no elements.

Complexity

- O(1)

---

## shape()

```cpp
const stratax::core::Shape& shape() const noexcept;
```

Returns vector shape metadata.

Complexity

- O(1)

---

## strides()

```cpp
const stratax::core::Strides& strides() const noexcept;
```

Returns vector stride metadata.

Complexity

- O(1)

---

## at()

```cpp
T& at(std::ptrdiff_t index);
const T& at(std::ptrdiff_t index) const;
```

Returns an element with bounds checking. Negative indices are normalized from the end.

Complexity

- O(1)

Throws

- `Exceptions::IndexError` if out of bounds

---

## front()

```cpp
T& front();
const T& front() const;
```

Returns first element.

Preconditions

- Vector must not be empty.

Complexity

- O(1)

---

## back()

```cpp
T& back();
const T& back() const;
```

Returns last element.

Preconditions

- Vector must not be empty.

Complexity

- O(1)

---

## data()

```cpp
[[nodiscard]] T* data() noexcept;
[[nodiscard]] const T* data() const noexcept;
```

Returns raw pointer to contiguous storage.

Complexity

- O(1)

---

## Iterators

```cpp
[[nodiscard]] T* begin() noexcept;
[[nodiscard]] const T* begin() const noexcept;
[[nodiscard]] const T* cbegin() const noexcept;
[[nodiscard]] T* end() noexcept;
[[nodiscard]] const T* end() const noexcept;
[[nodiscard]] const T* cend() const noexcept;
[[nodiscard]] std::reverse_iterator<T*> rbegin() noexcept;
[[nodiscard]] std::reverse_iterator<const T*> rbegin() const noexcept;
[[nodiscard]] std::reverse_iterator<const T*> crbegin() const noexcept;
[[nodiscard]] std::reverse_iterator<T*> rend() noexcept;
[[nodiscard]] std::reverse_iterator<const T*> rend() const noexcept;
[[nodiscard]] std::reverse_iterator<const T*> crend() const noexcept;
```

Provides forward and reverse iteration over contiguous elements.

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
void swap(Vector& other) noexcept;
```

Exchanges shape, strides, and storage with another vector.

Complexity

- O(1)

---

## Operators

## operator()

```cpp
T& operator()(std::size_t index) noexcept;
const T& operator()(std::size_t index) const noexcept;
```

Unchecked flat indexing.

Complexity

- O(1)

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

```cpp
T& at(std::ptrdiff_t index);
const T& at(std::ptrdiff_t index) const;
```

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Default construction | O(1) |
| Size/shape/fill/list construction | O(n) |
| Copy construction | O(n) |
| Move construction | O(1) |
| Copy assignment | O(n) |
| Move assignment | O(1) |
| Destruction | O(n) |
| `size()` / `rank()` / `empty()` | O(1) |
| `shape()` / `strides()` | O(1) |
| `operator()` / `operator[]` / `at()` | O(1) |
| `front()` / `back()` | O(1) |
| Iteration | O(n) |
| `fill()` | O(n) |
| `swap()` | O(1) |

---

## Examples

## Creating Vectors

```cpp
stratax::container::Vector<double> a;
stratax::container::Vector<double> b(5);
stratax::container::Vector<double> c{1.0, 2.0, 3.0};
```

---

## Accessing Elements

```cpp
stratax::container::Vector<double> v{10.0, 20.0, 30.0};

auto x = v[1];
auto y = v.at(-1);
```

---

## Iteration

```cpp
for (const auto& value : v)
{
    std::cout << value << '\n';
}
```

---

## Design Notes

`Vector<T>` keeps layout metadata (`Shape`, `Strides`) alongside contiguous data storage (`Buffer<T>`) so it can interoperate consistently with other Stratax containers.

Unchecked accessors are provided for performance-critical paths, while `at()` offers safe validated indexing including negative index normalization.

---

## Future Improvements

- Add slicing/view utilities for non-owning vector spans
- Add SIMD-specialized fill/copy paths where profitable
- Add optional small-vector optimization (if benchmark-justified)

---

## See Also

- Buffer
- Shape
- Strides
- Matrix
- Tensor
