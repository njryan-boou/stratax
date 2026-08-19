@page vector Vector

# Vector {#dev_vector}

Version: v0.2.0

Status: Complete

Header: `include/stratax/containers/Vector.hpp`

---

## Overview

`stratax::container::Vector<T>` is a one-dimensional owning array for types
that satisfy the `Numeric` concept. It derives from `core::ArrayBase<T>` and
uses the base class's contiguous `Buffer` and `Shape` metadata. Strides are
stored as a second `Shape` containing canonical row-major stride values.

A normally constructed vector always has rank one. This includes an empty
vector, whose shape is `{0}` and whose size is zero.

```cpp
stratax::container::Vector<double> values{1.0, 2.0, 3.0};

values[0] = 4.0; // unchecked access
values.at(-1);   // 3.0; negative indices count from the end
values.fill(2.0);
```

---

## Responsibilities

`Vector<T>` is responsible for:

- Enforcing a rank-one shape at construction
- Providing convenient size, fill-value, shape, and initializer-list constructors
- Exposing the common owning-array interface inherited from `ArrayBase<T>`
- Supporting constant-time member and argument-dependent `swap`

The inherited `ArrayBase<T>` interface provides:

- Contiguous storage ownership
- Shape and stride metadata
- Checked signed flat indexing through `at(index)`
- Unchecked flat indexing through `operator[]`
- Forward and reverse iteration
- `front()`, `back()`, `data()`, and `fill()`
- Copy and move value semantics

`Vector<T>` does not provide multidimensional indexing, broadcasting, or
rank-changing operations directly. Those operations belong to other
containers or algorithms.

---

## Representation and Invariants

```text
Vector<T>
└── core::ArrayBase<T>
    ├── core::Buffer<T> buffer_
    ├── core::Shape     shape_
    └── core::Shape     strides_
```

For every normally constructed vector:

- `rank() == 1`
- `shape() == Shape{size()}`
- `strides() == Shape{1}`
- `size() == shape().elements()`
- Elements occupy one contiguous memory range

As with other movable containers, a moved-from vector remains destructible and
assignable, but its previous contents and layout must not be relied upon.

---

## Type Aliases

`Vector<T>` republishes the complete container alias set from
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

These aliases allow generic contiguous-container code to use `Vector<T>`
without depending directly on `Buffer<T>`.

---

## Constructors

### Default Constructor

```cpp
Vector();
```

Constructs an empty rank-one vector with shape `{0}` and strides `{1}`.

Complexity: O(1).

### Size Constructor

```cpp
explicit Vector(size_type size);
```

Constructs a rank-one vector with `size` value-initialized elements. For
arithmetic types, value initialization produces zero.

Complexity: O(size).

May throw `std::bad_alloc` or an exception from `value_type` construction.

### Shape Constructor

```cpp
explicit Vector(const core::Shape& shape);
```

Constructs a value-initialized vector from `shape`. The shape must have exactly
one dimension; both `Shape{0}` and nonzero rank-one shapes are valid.

Complexity: O(shape.elements()).

Throws:

- `Exceptions::ShapeError` when `shape.rank() != 1`
- `std::bad_alloc` when allocation fails
- Any exception propagated from `value_type` construction

### Fill Constructor

```cpp
Vector(size_type size, const_reference value);
```

Constructs a rank-one vector containing `size` copies of `value`.

Complexity: O(size).

May throw `std::bad_alloc` or an exception from the `value_type` copy
constructor.

### Initializer-list Constructor

```cpp
Vector(std::initializer_list<value_type> list);
```

Copies the list elements into contiguous storage in their original order. An
empty list produces a rank-one vector with shape `{0}`.

Complexity: O(list.size()).

May throw `std::bad_alloc` or an exception from the `value_type` copy
constructor.

---

## Copy and Move Semantics

The compiler-generated special members use `ArrayBase<T>` semantics:

```cpp
Vector(const Vector&) = default;
Vector(Vector&&) = default;
Vector& operator=(const Vector&) = default;
Vector& operator=(Vector&&) = default;
~Vector() = default;
```

Copying duplicates the element storage and metadata. Moving transfers their
ownership. Copy operations are O(n), while moves are O(1), apart from
destroying any state replaced by move assignment.

---

## Inherited Container Interface

### Metadata

```cpp
[[nodiscard]] size_type size() const noexcept;
[[nodiscard]] bool empty() const noexcept;
[[nodiscard]] size_type rank() const noexcept;
[[nodiscard]] const core::Shape& shape() const noexcept;
[[nodiscard]] const core::Shape& strides() const noexcept;
```

All metadata queries are O(1).

### Element Access

```cpp
reference operator[](size_type index) noexcept;
const_reference operator[](size_type index) const noexcept;

reference at(difference_type index);
const_reference at(difference_type index) const;

reference front();
const_reference front() const;
reference back();
const_reference back() const;

[[nodiscard]] pointer data() noexcept;
[[nodiscard]] const_pointer data() const noexcept;
```

`operator[]` is unchecked and requires `index < size()`. `at()` checks its
argument and accepts indices in `[-size(), size())`; negative values count from
the end. It throws `Exceptions::IndexError` when the index is invalid.

`front()` and `back()` also throw `Exceptions::IndexError` for an empty vector.
All element-access operations are O(1).

`Vector` does not define `operator()`. Use `operator[]` for unchecked access or
`at()` for checked access.

### Iterators

```cpp
iterator begin() noexcept;
const_iterator begin() const noexcept;
const_iterator cbegin() const noexcept;
iterator end() noexcept;
const_iterator end() const noexcept;
const_iterator cend() const noexcept;

reverse_iterator rbegin() noexcept;
const_reverse_iterator rbegin() const noexcept;
const_reverse_iterator crbegin() const noexcept;
reverse_iterator rend() noexcept;
const_reverse_iterator rend() const noexcept;
const_reverse_iterator crend() const noexcept;
```

Each iterator lookup is O(1). Traversing the complete vector is O(size()).

### Fill

```cpp
void fill(const_reference value);
```

Assigns `value` to every element in O(size()).

---

## Swap

```cpp
void swap(Vector& other) noexcept;
friend void swap(Vector& lhs, Vector& rhs) noexcept;
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
| Size, fill, or list construction | O(n) |
| Shape construction | O(shape.elements()) |
| Copy construction or assignment | O(n) |
| Move construction | O(1) |
| Metadata query | O(1) |
| Element access | O(1) |
| Iterator acquisition | O(1) |
| Complete traversal | O(n) |
| `fill()` | O(n) |
| `swap()` | O(1) |

---

## Examples

### Construction and Metadata

```cpp
stratax::container::Vector<double> empty;
stratax::container::Vector<double> zeros(4);
stratax::container::Vector<double> filled(4, 2.5);
stratax::container::Vector<double> values{1.0, 2.0, 3.0};

empty.rank();   // 1
empty.shape();  // Shape{0}
values.size();  // 3
```

### Checked and Unchecked Access

```cpp
stratax::container::Vector<double> values{10.0, 20.0, 30.0};

values[1] = 25.0;  // unchecked
values.at(1);      // 25.0
values.at(-1);     // 30.0
```

### Iteration

```cpp
for (double& value : values)
{
    value *= 2.0;
}

for (auto it = values.crbegin(); it != values.crend(); ++it)
{
    std::cout << *it << '\n';
}
```

---

## Design Notes

Keeping storage and layout behavior in `ArrayBase<T>` gives `Vector`, `Matrix`,
and `Tensor` consistent ownership, access, and iterator semantics. `Vector<T>`
adds only the construction rules needed to preserve its rank-one abstraction.

---

## See Also

- @ref arraybase
- @ref buffer
- @ref shape
- @ref matrix
- @ref tensor
