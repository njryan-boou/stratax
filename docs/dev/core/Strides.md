@page strides Strides

# Strides {#dev_strides}

Version: v0.3.1

Status: Complete

Header: `include/stratax/core/Strides.hpp`

---

## Overview

`Strides` owns row-major stride metadata for a `Shape`. A stride is the number
of contiguous elements skipped when the index for one dimension increases by
one.

For a non-empty row-major shape, the final stride is `1`, and each preceding
stride is the product of the dimensions to its right:

```cpp
const stratax::core::Shape shape{2, 3, 4};
const stratax::core::Strides strides(shape);

// strides == (12, 4, 1)
const std::size_t offset =
    1 * strides[0] + 2 * strides[1] + 3 * strides[2]; // 23
```

Stride values are stored contiguously in `Buffer<std::size_t>`. Unlike
`Shape`, `Strides` exposes mutable references, pointers, and iterators.

---

## Responsibilities

`Strides` is responsible for:

- Computing canonical row-major strides from a shape
- Detecting stride multiplication overflow
- Owning stride values in contiguous storage
- Providing mutable and const element access
- Providing checked Python-style indexing
- Supporting forward and reverse iteration
- Supporting value comparison, swapping, and tuple-style formatting

It is not responsible for owning array elements, validating complete indices,
calculating multidimensional offsets, or performing numerical operations.

---

## Relationships

```text
Shape ──constructs──> Strides
                       └── buffer_ : Buffer<std::size_t>
```

Depends on:

- `Shape` for source dimensions
- `Buffer<std::size_t>` for owned contiguous storage
- `indexing::normalize_index` for checked signed indexing
- `Exceptions::DimensionError` and `Exceptions::IndexError`

Used by `ArrayBase` and the indexing, slicing, reshape, and printing layers.

---

## Invariants

- `rank()` equals the number of stored stride values.
- `empty()` is equivalent to `rank() == 0`.
- An empty shape produces empty strides and `data() == nullptr`.
- A non-empty canonical stride sequence ends in `1`.
- Each earlier canonical stride is the representable product of dimensions to
  its right.
- Construction throws rather than allowing unsigned stride overflow.
- Values may be changed after construction through mutable accessors.
- Equal instances have identical ranks and stride values.

Because values are mutable, changing them can make an instance no longer
represent the canonical row-major layout of its original shape.

---

## Public Types

```cpp
using value_type             = std::size_t;
using size_type              = std::size_t;
using difference_type        = std::ptrdiff_t;
using reference              = value_type&;
using const_reference        = const value_type&;
using pointer                = value_type*;
using const_pointer          = const value_type*;
using iterator               = Buffer<value_type>::iterator;
using const_iterator         = Buffer<value_type>::const_iterator;
using reverse_iterator       = Buffer<value_type>::reverse_iterator;
using const_reverse_iterator = Buffer<value_type>::const_reverse_iterator;
```

The iterator aliases provide contiguous random-access traversal.

---

## Construction and Lifetime

### Default construction

```cpp
Strides() noexcept;
```

Constructs an empty, rank-zero stride sequence with no allocation.

Complexity: O(1).

### Shape construction

```cpp
explicit Strides(const Shape& shape);
```

Computes row-major strides from right to left. An empty shape produces an empty
sequence; a rank-one shape produces `(1,)`.

Zero dimensions are supported. For example:

```text
Shape{0, 3, 4} -> (12, 4, 1)
Shape{2, 0, 4} -> (0, 4, 1)
Shape{2, 3, 0} -> (0, 0, 1)
```

Complexity: O(shape.rank()).

Throws:

- `std::bad_alloc` if stride storage cannot be allocated
- `Exceptions::DimensionError` if a stride product exceeds `std::size_t`

### Copy and move construction

```cpp
Strides(const Strides&) = default;
Strides(Strides&&) noexcept = default;
```

Copy construction creates independent storage in O(rank()) time. Move
construction transfers the allocation in O(1), leaves the source empty, and
does not move individual stride values.

### Destruction

```cpp
~Strides() = default;
```

Destroys the underlying stride buffer in O(rank()) time.

---

## Assignment

```cpp
Strides& operator=(const Strides&) = default;
Strides& operator=(Strides&&) noexcept = default;
```

Copy assignment performs a deep copy with the strong exception guarantee. Its
complexity is O(old rank + copied rank).

Move assignment destroys the destination's previous storage and transfers the
source allocation. Its complexity is O(old rank), followed by O(1) ownership
transfer. The source becomes empty. Self-copy and self-move assignment leave
the object unchanged.

---

## Metadata and Storage

### rank and empty

```cpp
[[nodiscard]] size_type rank() const noexcept;
[[nodiscard]] bool empty() const noexcept;
```

`rank()` returns the number of stride values. `empty()` is equivalent to
`rank() == 0`. Both operations are O(1).

### data

```cpp
[[nodiscard]] pointer data() noexcept;
[[nodiscard]] const_pointer data() const noexcept;
```

Returns the first stride's address, or `nullptr` when empty. The non-const
overload permits direct mutation. Pointers remain valid until the owning object
is assigned to, moved from, swapped, or destroyed.

Complexity: O(1).

---

## Element Access

### front and back

```cpp
[[nodiscard]] reference front();
[[nodiscard]] const_reference front() const;
[[nodiscard]] reference back();
[[nodiscard]] const_reference back() const;
```

Returns the first or final stride. The mutable overloads permit modification.
Each function throws `Exceptions::IndexError` when the sequence is empty.

Complexity: O(1).

### Unchecked access

```cpp
[[nodiscard]] const_reference operator[](size_type index) const noexcept;
```

Returns a read-only reference without checking bounds. Behavior is undefined
unless `index < rank()`.

Complexity: O(1).

### Checked signed access

```cpp
[[nodiscard]] const_reference at(difference_type index) const;
```

Accepts indices in `[-rank(), rank())`. Negative indices count backward from
the final stride:

```cpp
const Strides strides(Shape{2, 3, 4});

strides.at(0);  // 12
strides.at(-1); // 1
strides.at(-3); // 12
```

Throws `Exceptions::IndexError` when the index is outside the valid range,
including every index applied to an empty sequence.

Complexity: O(1).

---

## Iteration

```cpp
iterator begin() noexcept;
iterator end() noexcept;
const_iterator begin() const noexcept;
const_iterator end() const noexcept;
const_iterator cbegin() const noexcept;
const_iterator cend() const noexcept;

reverse_iterator rbegin() noexcept;
reverse_iterator rend() noexcept;
const_reverse_iterator rbegin() const noexcept;
const_reverse_iterator rend() const noexcept;
const_reverse_iterator crbegin() const noexcept;
const_reverse_iterator crend() const noexcept;
```

Non-const iterators permit stride mutation. Each accessor is O(1), while full
traversal is O(rank()). Every begin/end pair compares equal when empty.

---

## Comparison

```cpp
[[nodiscard]] bool operator==(const Strides& other) const noexcept;
```

Two instances compare equal when their ranks and corresponding stride values
are equal. In C++20, `operator!=` is rewritten from `operator==`.

Complexity: O(rank()) in the worst case; differing ranks return in O(1).

---

## Swap

```cpp
void swap(Strides& other) noexcept;
```

Exchanges storage without copying individual values. Pointers, references, and
iterators remain valid but now refer to values owned by the other object.
Swapping with an empty instance and self-swap are supported.

Complexity: O(1).

---

## Stream Output

```cpp
std::ostream& operator<<(std::ostream& os, const Strides& strides);
```

Writes tuple-style output and returns `os`:

```text
Strides{Shape{2, 3, 4}} -> (12, 4, 1)
Strides{Shape{5}}       -> (1,)
Strides{}               -> ()
```

Complexity: O(strides.rank()).

---

## Complexity Summary

| Operation | Complexity |
| --------- | ---------- |
| Default construction | O(1) |
| Shape construction | O(n) |
| Copy construction | O(n) |
| Move construction | O(1) |
| Copy assignment | O(old rank + copied rank) |
| Move assignment | O(old rank) |
| Destruction | O(n) |
| Metadata, individual access, or iterator creation | O(1) |
| Full traversal or comparison | O(n) |
| `swap()` | O(1) |
| Stream output | O(n) |

---

## Design Notes

`Strides` separates layout metadata from both shape metadata and array element
storage. This lets indexing code consume a simple contiguous stride sequence
without depending on a particular array container.

The class permits mutation because slicing and future layout transformations
may need non-canonical stride values. Construction from `Shape`, however,
always computes canonical row-major values or throws if they are not
representable.

---

## See Also

- @ref shape
- @ref buffer
- @ref arraybase
- @ref vector
- @ref matrix
- @ref tensor
