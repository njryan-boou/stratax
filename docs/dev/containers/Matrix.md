@page matrix Matrix

# Matrix {#dev_matrix}

Version: v0.2.0

Status: Complete

Header: `include/stratax/containers/Matrix.hpp`

---

## Overview

`stratax::container::Matrix<T>` is a two-dimensional owning array for types
that satisfy the `Numeric` concept. It derives from `core::ArrayBase<T>` and
stores elements contiguously in row-major order.

A normally constructed matrix always has rank two. Empty matrices retain that
rank and may have shape `{0, 0}`, `{0, n}`, or `{n, 0}`.

```cpp
stratax::container::Matrix<double> matrix{
    {1.0, 2.0, 3.0},
    {4.0, 5.0, 6.0}
};

matrix(1, 2);     // 6.0; unchecked access
matrix.at(-1, 0); // 4.0; checked access
```

---

## Responsibilities

`Matrix<T>` is responsible for:

- Enforcing a rank-two shape at construction
- Validating that nested initializer lists are rectangular
- Exposing row and column counts
- Providing unchecked `operator()(row, col)` access
- Providing checked signed `at(row, col)` access
- Retaining the flat owning-container API inherited from `ArrayBase<T>`
- Supporting constant-time member and argument-dependent `swap`

`Matrix<T>` does not directly implement broadcasting, reshaping, row/column
views, or high-level numerical algorithms.

---

## Representation and Invariants

```text
Matrix<T>
└── core::ArrayBase<T>
    ├── core::Buffer<T> buffer_
    ├── core::Shape     shape_
    └── core::Shape     strides_
```

For every normally constructed matrix:

- `rank() == 2`
- `shape() == Shape{rows(), cols()}`
- `size() == rows() * cols()`
- The row-major strides are `{cols(), 1}` for nonzero column counts
- `operator[](row * cols() + col)` addresses the same element as
  `operator()(row, col)` for valid indices
- Elements occupy one contiguous memory range

Element-count and stride multiplication are checked during construction. A
moved-from matrix remains destructible and assignable, but its previous
contents and layout must not be relied upon.

---

## Type Aliases

`Matrix<T>` republishes the complete container alias set from
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
Matrix();
```

Constructs an empty rank-two matrix with shape `{0, 0}`.

Complexity: O(1).

### Dimensions Constructor

```cpp
Matrix(size_type rows, size_type cols);
```

Constructs a matrix containing `rows * cols` value-initialized elements. For
arithmetic types, value initialization produces zero.

Complexity: O(rows * cols).

Throws:

- `Exceptions::DimensionError` if the element count or a stride overflows
- `std::bad_alloc` if allocation fails
- Any exception propagated from `value_type` construction

### Shape Constructor

```cpp
explicit Matrix(const core::Shape& shape);
```

Constructs a value-initialized matrix from a rank-two shape. Shapes containing
a zero dimension are valid.

Complexity: O(shape.elements()).

Throws:

- `Exceptions::ShapeError` when `shape.rank() != 2`
- `Exceptions::DimensionError` if the element count or a stride overflows
- `std::bad_alloc` if allocation fails
- Any exception propagated from `value_type` construction

### Fill Constructor

```cpp
Matrix(size_type rows, size_type cols, const_reference value);
```

Constructs a matrix containing `rows * cols` copies of `value`.

Complexity: O(rows * cols).

It has the same overflow and allocation failure conditions as the dimensions
constructor and may propagate exceptions from the `value_type` copy
constructor.

### Nested Initializer-list Constructor

```cpp
Matrix(std::initializer_list<std::initializer_list<value_type>> list);
```

Copies rows into contiguous row-major storage. Every row must have the same
length. An empty outer list creates shape `{0, 0}`; a non-empty collection of
empty rows creates shape `{rows, 0}`.

Complexity: O(rows * cols), with an additional O(rows) validation pass.

Throws:

- `Exceptions::ShapeError` if row lengths differ
- `Exceptions::DimensionError` if the element count or a stride overflows
- `std::bad_alloc` if allocation fails
- Any exception propagated from `value_type` construction or assignment

---

## Copy and Move Semantics

The compiler-generated special members use `ArrayBase<T>` semantics:

```cpp
Matrix(const Matrix&) = default;
Matrix(Matrix&&) = default;
Matrix& operator=(const Matrix&) = default;
Matrix& operator=(Matrix&&) = default;
~Matrix() = default;
```

Copying duplicates element storage and metadata. Moving transfers their
ownership. Copy operations are O(n), while move construction is O(1).

---

## Matrix Metadata

```cpp
[[nodiscard]] size_type rows() const noexcept;
[[nodiscard]] size_type cols() const noexcept;
```

`rows()` returns `shape()[0]`, and `cols()` returns `shape()[1]`. Both
operations are O(1).

The inherited metadata interface is also available:

```cpp
[[nodiscard]] size_type size() const noexcept;
[[nodiscard]] bool empty() const noexcept;
[[nodiscard]] size_type rank() const noexcept;
[[nodiscard]] const core::Shape& shape() const noexcept;
[[nodiscard]] const core::Shape& strides() const noexcept;
```

---

## Element Access

### Unchecked Two-dimensional Access

```cpp
reference operator()(size_type row, size_type col);
const_reference operator()(size_type row, size_type col) const;
```

Computes the row-major offset `row * cols() + col` without checking either
index.

Preconditions:

- `row < rows()`
- `col < cols()`

Complexity: O(1).

### Checked Two-dimensional Access

```cpp
reference at(difference_type row, difference_type col);
const_reference at(difference_type row, difference_type col) const;
```

Checks each component independently. Valid row indices are
`[-rows(), rows())`, and valid column indices are `[-cols(), cols())`.
Negative components count backward from the corresponding dimension.

Complexity: O(1).

Throws `Exceptions::IndexError` if either component is invalid.

### Inherited Flat Access

```cpp
reference operator[](size_type index) noexcept;
const_reference operator[](size_type index) const noexcept;

reference at(difference_type index);
const_reference at(difference_type index) const;
```

`operator[]` is unchecked. The inherited one-argument `at(index)` overload is
explicitly retained with a using-declaration and supports negative flat
indices.

The inherited `front()`, `back()`, and `data()` accessors are also
available. `front()` and `back()` throw `Exceptions::IndexError` when the
matrix is empty.

All individual element-access operations are O(1).

---

## Iteration and Fill

`begin()`, `end()`, their const variants, and all reverse iterator variants
traverse the flat row-major sequence. Acquiring an iterator is O(1), and
traversing all elements is O(size()).

```cpp
for (auto& value : matrix)
{
    value *= 2;
}
```

`fill(const_reference value)` assigns `value` to every element in O(size()).

---

## Swap

```cpp
void swap(Matrix& other) noexcept;
friend void swap(Matrix& lhs, Matrix& rhs) noexcept;
```

Both overloads exchange the buffer, shape, and strides in O(1). The non-member
overload supports argument-dependent lookup:

```cpp
using std::swap;
swap(lhs, rhs);
```

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Default construction | O(1) |
| Dimension, fill, shape, or list construction | O(rows * cols) |
| Copy construction or assignment | O(n) |
| Move construction | O(1) |
| Metadata query | O(1) |
| Flat or two-dimensional element access | O(1) |
| Iterator acquisition | O(1) |
| Complete traversal | O(n) |
| `fill()` | O(n) |
| `swap()` | O(1) |

---

## Examples

### Construction

```cpp
stratax::container::Matrix<double> empty;
stratax::container::Matrix<double> zeros(2, 3);
stratax::container::Matrix<double> filled(2, 2, 1.5);
stratax::container::Matrix<double> values{
    {1.0, 2.0},
    {3.0, 4.0}
};
```

### Checked and Unchecked Access

```cpp
stratax::container::Matrix<double> values{
    {10.0, 20.0},
    {30.0, 40.0}
};

values(1, 0) = 35.0; // unchecked row/column access
values.at(1, 0);     // 35.0
values.at(-1, -1);   // 40.0
values.at(-1);       // 40.0 through inherited flat access
```

### Row-major Traversal

```cpp
for (const double value : values)
{
    std::cout << value << '\n';
}
```

The loop visits `10.0`, `20.0`, `35.0`, then `40.0`.

---

## Design Notes

Keeping ownership and layout behavior in `ArrayBase<T>` gives `Vector`,
`Matrix`, and `Tensor` consistent flat access and iterator semantics.
`Matrix<T>` adds only rank-two validation, dimension queries, rectangular
initializer handling, and two-dimensional indexing.

---

## See Also

- @ref arraybase
- @ref buffer
- @ref shape
- @ref vector
- @ref tensor
