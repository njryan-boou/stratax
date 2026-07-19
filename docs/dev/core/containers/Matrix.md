# Matrix\\<T\\>

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/containers/Matrix.hpp`

---

## Overview

`Matrix<T>` is Stratax's rank-2 owning array container for numeric types.

It stores values contiguously in row-major order and pairs storage with `Shape` and `Strides` metadata for consistent container behavior across the library.

---

## Responsibilities

The `Matrix` class is responsible for:

- Owning contiguous rank-2 element storage
- Exposing row/column shape and stride metadata
- Providing checked and unchecked element access
- Providing iterator access over contiguous row-major storage
- Supporting copy/move ownership semantics

The `Matrix` class is **not** responsible for:

- Rank-1 or rank-N container semantics
- Broadcasting policy decisions
- High-level numerical algorithms
- Tensor reshaping outside matrix-specific APIs

---

## Relationships

```text
Matrix<T>
│
├── shape_   : core::Shape
├── strides_ : core::Strides
└── buffer_  : core::Buffer<T>
```

Depends on:

- Buffer
- Shape
- Strides
- Validation helpers (`validation::checked_multiply`, `validation::normalize_index`, `validation::require_index`, `validation::require_rank`)
- Exceptions (`Exceptions::IndexError`, `Exceptions::DimensionError`, `Exceptions::ShapeError`)

Used by:

- Tensor conversion and interop paths
- Generic Stratax container utilities

Related classes:

- Shape
- Strides
- Buffer
- Vector
- Tensor

---

## Internal Data

| Member | Description |
| ------- | ----------- |
| `core::Shape shape_` | Matrix shape metadata (rows, cols) |
| `core::Strides strides_` | Row-major stride metadata |
| `core::Buffer<T> buffer_` | Contiguous row-major element storage |

---

## Invariants

The following conditions are always true:

- `rank()` is 2 for constructed matrices.
- `size()` equals `rows() * cols()` (with overflow guarded at construction).
- `data()` points to row-major contiguous storage when non-empty.
- `at(row, col)` validates bounds and supports negative indexing.
- `operator[](flat)` is unchecked flat access.

---

## Public Interface

## Constructors

### Default Constructor

```cpp
Matrix();
```

Constructs an empty matrix with shape `(0, 0)`.

Complexity

- O(1)

Throws

- None

---

### Rows/Cols Constructor

```cpp
Matrix(std::size_t rows, std::size_t cols);
```

Constructs a matrix with default-initialized elements.

Complexity

- O(n)

Throws

- `std::bad_alloc`
- `Exceptions::DimensionError` on size overflow

---

### Shape Constructor

```cpp
explicit Matrix(const core::Shape& shape);
```

Constructs from a validated rank-2 shape.

Complexity

- O(n)

Throws

- `Exceptions::DimensionError` if shape rank is not 2
- `std::bad_alloc`
- `Exceptions::DimensionError` on size overflow

---

### Fill Constructor

```cpp
Matrix(std::size_t rows, std::size_t cols, const T& value);
```

Constructs and fills every element with `value`.

Complexity

- O(n)

Throws

- `std::bad_alloc`
- `Exceptions::DimensionError` on size overflow

---

### Nested Initializer List Constructor

```cpp
Matrix(std::initializer_list<std::initializer_list<T>> list);
```

Constructs from nested row lists in row-major order.

Complexity

- O(n)

Throws

- `Exceptions::ShapeError` if rows are not rectangular
- `std::bad_alloc`
- `Exceptions::DimensionError` on size overflow

---

### Copy Constructor

```cpp
Matrix(const Matrix&) = default;
```

Complexity

- O(n)

---

### Move Constructor

```cpp
Matrix(Matrix&&) noexcept = default;
```

Complexity

- O(1)

---

### Destructor

```cpp
~Matrix() = default;
```

Complexity

- O(n)

---

## Assignment Operators

## Copy Assignment

```cpp
Matrix& operator=(const Matrix&) = default;
```

Complexity

- O(n)

---

## Move Assignment

```cpp
Matrix& operator=(Matrix&&) noexcept = default;
```

Complexity

- O(1)

---

## Methods

## size()

```cpp
[[nodiscard]] std::size_t size() const noexcept;
```

Returns total number of stored elements.

Complexity

- O(1)

---

## empty()

```cpp
[[nodiscard]] bool empty() const noexcept;
```

Returns whether no elements are stored.

Complexity

- O(1)

---

## rows() / cols()

```cpp
[[nodiscard]] std::size_t rows() const noexcept;
[[nodiscard]] std::size_t cols() const noexcept;
```

Returns matrix dimensions.

Complexity

- O(1)

---

## shape() / strides() / rank()

```cpp
const stratax::core::Shape& shape() const noexcept;
const stratax::core::Strides& strides() const noexcept;
[[nodiscard]] std::size_t rank() const noexcept;
```

Returns matrix metadata.

Complexity

- O(1)

---

## at()

```cpp
T& at(std::ptrdiff_t row, std::ptrdiff_t col);
const T& at(std::ptrdiff_t row, std::ptrdiff_t col) const;
```

Returns element with bounds checking and negative-index normalization.

Complexity

- O(1)

Throws

- `Exceptions::IndexError` if out of bounds

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

- Matrix must not be empty.

Complexity

- O(1)

---

## data()

```cpp
[[nodiscard]] T* data() noexcept;
[[nodiscard]] const T* data() const noexcept;
```

Returns pointer to contiguous row-major storage.

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

Provides forward and reverse iteration over flat row-major storage.

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
void swap(Matrix& other) noexcept;
```

Exchanges metadata and storage with another matrix.

Complexity

- O(1)

---

## Operators

## operator()(row, col)

```cpp
T& operator()(std::size_t row, std::size_t col);
const T& operator()(std::size_t row, std::size_t col) const;
```

Returns element by row/column with bounds checking.

Complexity

- O(1)

Throws

- `Exceptions::IndexError` if row or column is out of bounds

---

## operator[]

```cpp
T& operator[](std::size_t index) noexcept;
const T& operator[](std::size_t index) const noexcept;
```

Unchecked flat indexing in row-major order.

Complexity

- O(1)

See Also

```cpp
T& at(std::ptrdiff_t row, std::ptrdiff_t col);
const T& at(std::ptrdiff_t row, std::ptrdiff_t col) const;
```

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Default construction | O(1) |
| Rows/cols/shape/fill/list construction | O(n) |
| Copy construction | O(n) |
| Move construction | O(1) |
| Copy assignment | O(n) |
| Move assignment | O(1) |
| Destruction | O(n) |
| `size()` / `rows()` / `cols()` / `rank()` / `empty()` | O(1) |
| `shape()` / `strides()` | O(1) |
| `operator()` / `operator[]` / `at()` | O(1) |
| `front()` / `back()` | O(1) |
| Iteration | O(n) |
| `fill()` | O(n) |
| `swap()` | O(1) |

---

## Examples

## Creating Matrices

```cpp
stratax::container::Matrix<double> a;
stratax::container::Matrix<double> b(2, 3);
stratax::container::Matrix<double> c(2, 2, 1.5);
stratax::container::Matrix<double> d{{1.0, 2.0}, {3.0, 4.0}};
```

---

## Accessing Elements

```cpp
stratax::container::Matrix<double> m{{10.0, 20.0}, {30.0, 40.0}};

auto x = m(1, 0);
auto y = m.at(-1, -1);
```

---

## Iteration

```cpp
for (const auto& value : m)
{
    std::cout << value << '\n';
}
```

---

## Design Notes

`Matrix<T>` keeps shape/stride metadata explicit while storing values in one contiguous row-major `Buffer<T>`. This mirrors `Vector` and `Tensor` internals and simplifies shared indexing and conversion logic.

Checked APIs (`operator()(row, col)`, `at`) coexist with unchecked flat access (`operator[]`) to balance safety and performance.

---

## Future Improvements

- Add row/column view APIs
- Add slicing/submatrix view support
- Add specialized SIMD kernels for common elementwise operations

---

## See Also

- Buffer
- Shape
- Strides
- Vector
- Tensor
