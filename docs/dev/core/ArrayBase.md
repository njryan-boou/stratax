@page arraybase ArrayBase

# ArrayBase {#dev_arraybase}

Version: v0.3.1

Status: Complete

Header: `include/stratax/core/ArrayBase.hpp`

---

## Overview

`ArrayBase<T>` is the shared owning base for Stratax's `Vector<T>`,
`Matrix<T>`, and `Tensor<T>` containers. It keeps three pieces of state
together:

```text
ArrayBase<T>
├── buffer_  : Buffer<T>  — contiguous element storage
├── shape_   : Shape      — logical dimensions
└── strides_ : Shape      — row-major layout metadata
```

The class supplies the common one-dimensional container interface and
protected offset helpers. Derived containers remain responsible for rank
constraints, constructors, and public multidimensional indexing APIs.

`ArrayBase` cannot be constructed directly by users because its constructors
are protected.

---

## Responsibilities

`ArrayBase` is responsible for:

- Owning contiguous element storage
- Keeping storage, shape, and stride metadata synchronized
- Providing common metadata and flat element access
- Providing mutable and const contiguous iterators
- Providing shared `fill()` and `swap()` operations
- Constructing storage from a shape or adopting an existing buffer
- Normalizing checked multidimensional indices for derived containers
- Computing unchecked flat offsets for already-normalized indices

It is not responsible for validating container-specific rank requirements,
defining multidimensional public APIs, numerical operations, broadcasting, or
formatting.

---

## Invariants

For every successfully constructed object:

- `size() == shape().elements()`
- `rank() == shape().rank()`
- `rank() == strides().rank()`
- `strides()` represents the row-major layout of `shape()`
- Non-empty elements occupy `[data(), data() + size())`
- Shape and stride metadata are exposed read-only through the public API

`swap()` exchanges all three state members together, preserving these
relationships.

---

## Public Types

The aliases mirror `Buffer<T>`:

```cpp
using value_type             = Buffer<T>::value_type;
using size_type              = Buffer<value_type>::size_type;
using difference_type        = Buffer<value_type>::difference_type;
using reference              = Buffer<value_type>::reference;
using const_reference        = Buffer<value_type>::const_reference;
using pointer                = Buffer<value_type>::pointer;
using const_pointer          = Buffer<value_type>::const_pointer;
using iterator               = Buffer<value_type>::iterator;
using const_iterator         = Buffer<value_type>::const_iterator;
using reverse_iterator       = Buffer<value_type>::reverse_iterator;
using const_reverse_iterator = Buffer<value_type>::const_reverse_iterator;
```

The pointer-based iterator types provide contiguous random-access traversal.

---

## Metadata

```cpp
[[nodiscard]] size_type size() const noexcept;
[[nodiscard]] bool empty() const noexcept;
[[nodiscard]] size_type rank() const noexcept;
[[nodiscard]] const Shape& shape() const noexcept;
[[nodiscard]] const Shape& strides() const noexcept;
```

- `size()` returns the number of stored elements.
- `empty()` is equivalent to `size() == 0`.
- `rank()` returns the number of logical dimensions.
- `shape()` and `strides()` return read-only metadata references.

Each operation is O(1).

---

## Element Storage and Access

### data

```cpp
[[nodiscard]] pointer data() noexcept;
[[nodiscard]] const_pointer data() const noexcept;
```

Returns the first element's address, or `nullptr` when empty. Pointers remain
valid until the object is assigned to, moved from, swapped, or destroyed.

Complexity: O(1).

### front and back

```cpp
reference front();
const_reference front() const;
reference back();
const_reference back() const;
```

Returns the first or final flat element. Each operation throws
`Exceptions::IndexError` when storage is empty.

Complexity: O(1).

### Unchecked flat access

```cpp
reference operator[](size_type index) noexcept;
const_reference operator[](size_type index) const noexcept;
```

Returns a flat element without bounds checking. Behavior is undefined unless
`index < size()`.

Complexity: O(1).

### Checked flat access

```cpp
reference at(difference_type index);
const_reference at(difference_type index) const;
```

Accepts indices in `[-size(), size())`. Negative indices count backward from
the final flat element. An out-of-range index, including any index into empty
storage, throws `Exceptions::IndexError`.

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

Each accessor is O(1). Traversing the full element range is O(size()). Begin
and end pairs compare equal when storage is empty.

---

## Modifiers

### fill

```cpp
void fill(const_reference value);
```

Copy-assigns `value` to every element from first to last. If assignment throws,
earlier elements retain their new values and later elements retain their old
values.

Complexity: O(size()).

### swap

```cpp
void swap(ArrayBase& other) noexcept;
```

Exchanges the buffer, shape, and strides in constant time. Pointers,
references, and iterators remain valid but refer to elements owned by the
other object after the exchange.

Complexity: O(1).

---

## Protected Construction

### Value-initialized storage

```cpp
explicit ArrayBase(const Shape& shape);
```

Allocates `shape.elements()` elements initialized from `value_type{}`, copies
the shape, and computes its row-major strides.

Complexity: O(shape.elements() + shape.rank()).

### Filled storage

```cpp
ArrayBase(const Shape& shape, const_reference value);
```

Allocates `shape.elements()` copies of `value`, copies the shape, and computes
its row-major strides.

Complexity: O(shape.elements() + shape.rank()).

Both allocating constructors may throw:

- `Exceptions::DimensionError` for an unrepresentable element count or stride
- `std::bad_alloc` if storage or metadata allocation fails
- An exception from element construction

### Adopted storage

```cpp
ArrayBase(const Shape& shape, Buffer<value_type>&& buffer);
```

Transfers ownership of `buffer`, copies the shape, computes strides, and then
requires `buffer.size() == shape.elements()`. A mismatch throws
`Exceptions::ShapeError`; the adopted buffer is released during unwinding.

Complexity: O(shape.rank()) plus O(1) ownership transfer.

Metadata allocation or an unrepresentable element/stride calculation can also
throw before construction completes.

---

## Protected Indexing Helpers

### normalized_flat_offset

```cpp
template<typename IndexContainer>
size_type normalized_flat_offset(
    const IndexContainer& raw_indices,
    const char* rank_mismatch_message =
        "Multi-index rank must match array rank.",
    const char* component_oob_message = nullptr) const;
```

Converts one signed index per dimension into a row-major flat offset. Negative
components count backward within their corresponding dimensions.

The index container must provide `size()` and indexed access. A rank mismatch
throws `Exceptions::IndexError` using `rank_mismatch_message`. An invalid
component also throws `IndexError`; when `component_oob_message` is non-null,
that message replaces the lower-level normalization message.

Complexity: O(rank()).

### flat_offset

```cpp
template<typename IndexContainer>
size_type flat_offset(const IndexContainer& indices) const;
```

Computes an unchecked row-major offset. The container must be iterable and
must contain exactly `rank()` already-normalized, in-range indices.

Violating the rank requirement may advance past the stride range and cause
undefined behavior. Out-of-range components can produce an invalid flat
offset.

Complexity: O(rank()).

---

## Copy and Move Semantics

`ArrayBase` uses compiler-generated copy and move operations:

- Copying deeply copies element storage and metadata.
- Moving transfers owned buffers and leaves the source members empty.
- Copy assignment inherits `Buffer`'s strong exception guarantee for each
  member assignment, but the compiler-generated member-by-member operation is
  not transactional across all three members.

Concrete containers inherit these operations unless they define their own.

---

## Complexity Summary

| Operation | Complexity |
| --------- | ---------- |
| Metadata, individual access, iterator creation | O(1) |
| Full traversal or `fill()` | O(size()) |
| `swap()` | O(1) |
| Shape-based construction | O(elements + rank) |
| Buffer-adopting construction | O(rank) |
| Checked/unchecked multidimensional offset | O(rank) |
| Copy construction | O(size + rank) |
| Move construction | O(1) |

---

## Design Notes

Keeping the common flat interface in one base class gives all owning arrays the
same storage and iterator semantics. Container-specific classes stay small and
focus on shape restrictions and multidimensional access.

The checked and unchecked offset helpers are intentionally separate. Public
`at()` implementations use normalized checking, while `operator()` paths can
use `flat_offset()` when their contract already requires valid indices.

---

## See Also

- @ref buffer
- @ref shape
- @ref vector
- @ref matrix
- @ref tensor
