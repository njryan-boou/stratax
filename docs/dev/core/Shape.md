@page shape Shape

# Shape {#dev_shape}

Version: v0.3.1

Status: Complete

Header: `include/stratax/core/Shape.hpp`

---

## Overview

`Shape` stores the dimensions of a multidimensional array in
outermost-to-innermost order. It reports the dimensional rank, computes the
total element count with overflow checking, supports checked Python-style
indexing, and exposes read-only contiguous iterators.

Dimension storage is owned by a `Buffer<std::size_t>`. `Shape` contains only
layout metadata; it does not own an array's numeric elements.

```cpp
const stratax::core::Shape shape{2, 3, 4};

shape.rank();     // 3
shape.elements(); // 24
shape[0];         // 2: unchecked access
shape.at(-1);     // 4: checked access from the end
```

---

## Responsibilities

`Shape` is responsible for:

- Owning dimension lengths in contiguous storage
- Reporting rank and rank-zero state
- Computing element counts without unsigned overflow
- Providing checked and unchecked read-only dimension access
- Supporting forward and reverse dimension iteration
- Supporting value comparison, swapping, and tuple-style formatting

It is not responsible for array element storage, stride calculation,
multidimensional offset calculation, or numerical operations.

---

## Relationships

```text
Shape
└── dims_ : Buffer<std::size_t>
    └── dimensions in outermost-to-innermost order
```

Depends on:

- `Buffer<std::size_t>` for owned contiguous storage
- `indexing::normalize_index` for checked signed indexing
- `Exceptions::IndexError` and `Exceptions::DimensionError`

Used by:

- `Strides`
- `ArrayBase`
- `Vector`, `Matrix`, and `Tensor`
- Indexing, slicing, reshape, and creation algorithms

---

## Invariants

- `rank()` equals the number of stored dimensions.
- `empty()` is equivalent to `rank() == 0`.
- Dimensions are stored in outermost-to-innermost order.
- Dimensions are exposed only through const references and const iterators.
- Zero-valued dimensions are permitted.
- Equal shapes have the same rank and identical dimensions in the same order.
- A moved-from shape is empty because its underlying buffer transfers ownership.

---

## Public Types

```cpp
using const_iterator = Buffer<std::size_t>::const_iterator;
using const_reverse_iterator =
    Buffer<std::size_t>::const_reverse_iterator;
```

Both aliases provide read-only random-access traversal. `Shape` intentionally
does not expose mutable dimension iterators.

---

## Construction and Lifetime

### Default construction

```cpp
Shape() noexcept;
```

Constructs an empty, rank-zero shape. Its element count is zero.

Complexity: O(1).

### Initializer-list construction

```cpp
Shape(std::initializer_list<std::size_t> dims);
```

Copies dimensions in list order. An empty list constructs a rank-zero shape,
and zero-valued dimensions are accepted.

Complexity: O(dims.size()).

Throws `std::bad_alloc` if dimension storage cannot be allocated.

### Vector construction

```cpp
Shape(const std::vector<std::size_t>& dims);
```

Copies dimensions from a vector while preserving their order. An empty vector
constructs a rank-zero shape.

Complexity: O(dims.size()).

Throws `std::bad_alloc` if dimension storage cannot be allocated.

### Copy and move construction

```cpp
Shape(const Shape&) = default;
Shape(Shape&&) noexcept = default;
```

Copy construction creates independent dimension storage in O(rank()) time.
Move construction transfers storage in O(1) and leaves the source empty.

### Destruction

```cpp
~Shape() = default;
```

Destroys the underlying dimension buffer.

Complexity: O(rank()).

---

## Assignment

```cpp
Shape& operator=(const Shape&) = default;
Shape& operator=(Shape&&) noexcept = default;
```

Copy assignment performs a deep copy and inherits `Buffer`'s strong exception
guarantee. Its complexity is O(old rank + copied rank).

Move assignment destroys the destination's previous dimension storage and
then transfers ownership. Its complexity is O(old rank), with O(1) ownership
transfer. The source becomes empty. Self-copy and self-move assignment leave
the shape unchanged.

---

## Metadata

### rank

```cpp
[[nodiscard]] std::size_t rank() const noexcept;
```

Returns the number of dimensions.

Complexity: O(1).

### empty

```cpp
[[nodiscard]] bool empty() const noexcept;
```

Returns `true` when no dimensions are stored. A shape containing a zero
dimension is not empty because it still has nonzero rank.

Complexity: O(1).

### elements

```cpp
[[nodiscard]] std::size_t elements() const;
```

Returns the product of the dimensions. A rank-zero shape or a shape with a
zero-valued dimension has zero elements. Each multiplication is checked before
it is performed.

Complexity: O(rank()).

Throws `Exceptions::DimensionError` if a nonzero product cannot be represented
by `std::size_t`.

```cpp
Shape{2, 3, 4}.elements(); // 24
Shape{2, 0, 4}.elements(); // 0
```

---

## Dimension Access

### Unchecked access

```cpp
[[nodiscard]] const std::size_t&
operator[](std::size_t index) const noexcept;
```

Returns the dimension at a zero-based index without checking bounds. Behavior
is undefined unless `index < rank()`.

Complexity: O(1).

### Checked signed access

```cpp
[[nodiscard]] const std::size_t& at(std::ptrdiff_t index) const;
```

Accepts indices in `[-rank(), rank())`. Non-negative indices count from the
first dimension, while negative indices count backward from the last:

```cpp
const Shape shape{2, 3, 4};

shape.at(0);  // 2
shape.at(-1); // 4
shape.at(-3); // 2
```

Complexity: O(1).

Throws `Exceptions::IndexError` for an out-of-range index, including every
index applied to an empty shape.

---

## Iteration

```cpp
const_iterator begin() const noexcept;
const_iterator end() const noexcept;
const_iterator cbegin() const noexcept;
const_iterator cend() const noexcept;

const_reverse_iterator rbegin() const noexcept;
const_reverse_iterator rend() const noexcept;
const_reverse_iterator crbegin() const noexcept;
const_reverse_iterator crend() const noexcept;
```

Each accessor is O(1). Traversing every dimension is O(rank()). For an empty
shape, each begin/end pair compares equal.

```cpp
for (std::size_t dimension : Shape{2, 3, 4}) {
    // Visits 2, then 3, then 4.
}
```

---

## Comparison

```cpp
[[nodiscard]] bool operator==(const Shape& other) const noexcept;
```

Two shapes compare equal when their ranks and corresponding dimensions are
equal. In C++20, `operator!=` is rewritten from `operator==`.

Complexity: O(rank()) in the worst case; differing ranks return in O(1).

---

## Swap

```cpp
void swap(Shape& other) noexcept;
```

Exchanges dimension storage without copying individual values. Iterators and
references continue to refer to the same dimensions, which become owned by the
other shape. Swapping with an empty shape and self-swap are supported.

Complexity: O(1).

---

## Stream Output

```cpp
std::ostream& operator<<(std::ostream& os, const Shape& shape);
```

Writes tuple-style output and returns `os`:

```text
Shape{2, 3, 4} -> (2, 3, 4)
Shape{5}       -> (5,)
Shape{}        -> ()
```

Complexity: O(shape.rank()).

---

## Complexity Summary

| Operation | Complexity |
| --------- | ---------- |
| Default construction | O(1) |
| List/vector construction | O(n) |
| Copy construction | O(n) |
| Move construction | O(1) |
| Copy assignment | O(old rank + copied rank) |
| Move assignment | O(old rank) |
| Destruction | O(n) |
| `elements()` | O(n) |
| `rank()`, `empty()`, and individual access | O(1) |
| Iterator creation | O(1) |
| Full traversal or comparison | O(n) |
| `swap()` | O(1) |
| Stream output | O(n) |

---

## Design Notes

`Shape` is deliberately immutable through its public access API. To represent
different dimensions, construct or assign another shape. This protects the
relationship between shape metadata and the strides and storage owned by
higher-level containers.

The class uses a dynamic buffer rather than a fixed-rank representation so one
type can represent vectors, matrices, tensors, rank-zero shapes, and arbitrary
higher-dimensional layouts.

---

## See Also

- @ref buffer
- @ref strides
- @ref arraybase
- @ref vector
- @ref matrix
- @ref tensor
