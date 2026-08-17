@page buffer Buffer

# Buffer {#dev_buffer}

Version: v0.3.1

Status: Complete

Header: `include/stratax/core/Buffer.hpp`

---

## Overview

`Buffer<T, Alignment>` is a fixed-size owner of aligned, contiguous element
storage. It allocates raw storage, constructs every element, and destroys the
elements before releasing the allocation. Its size cannot change after
construction, although its elements remain mutable.

The default alignment is `config::default_alignment`. A custom alignment must
be a power of two and at least `alignof(T)`.

`Buffer` is the storage primitive used directly by `ArrayBase`, `Shape`, and
`Strides`, and therefore indirectly by Stratax's array containers.

```cpp
stratax::core::Buffer<float, 64> samples{1.0F, 2.0F, 3.0F};

samples[1] = 5.0F;
std::fill(samples.begin(), samples.end(), 4.0F);

auto copy = samples;   // Deep copy with an independent allocation.
copy.front() = 9.0F;   // samples remains unchanged.
```

---

## Responsibilities

`Buffer` is responsible for:

- Allocating and releasing aligned contiguous storage
- Managing element lifetime through RAII
- Providing deep-copy and ownership-transferring move semantics
- Providing mutable and const element access
- Providing contiguous random-access iterators
- Preserving a fixed element count between assignments

It is not responsible for resizing, multidimensional metadata, checked
`operator[]` access, or numerical operations.

---

## Template Parameters

```cpp
template<typename T, std::size_t Alignment = config::default_alignment>
class Buffer;
```

| Parameter | Description |
| --------- | ----------- |
| `T` | Stored element type |
| `Alignment` | Allocation alignment in bytes; a power of two no smaller than `alignof(T)` |

Invalid alignment values fail the class's compile-time assertions.

---

## Invariants and Ownership

- `data()` is `nullptr` exactly when the buffer is empty.
- A non-empty buffer owns `size()` live, contiguous elements.
- Every owned allocation uses `alignment()`-byte alignment.
- Copy operations allocate independent storage and copy every element.
- Move operations transfer the allocation and leave the source empty.
- Destruction destroys all live elements before releasing storage.
- The element count changes only through copy or move assignment.

Pointers, references, and iterators remain valid until the owning buffer is
assigned to, moved from, swapped, or destroyed. After `swap`, they still refer
to the same elements, but those elements are owned by the other buffer.

---

## Public Types

```cpp
using value_type             = T;
using size_type              = std::size_t;
using difference_type        = std::ptrdiff_t;
using reference              = T&;
using const_reference        = const T&;
using pointer                = T*;
using const_pointer          = const T*;
using iterator               = pointer;
using const_iterator         = const_pointer;
using reverse_iterator       = std::reverse_iterator<iterator>;
using const_reverse_iterator = std::reverse_iterator<const_iterator>;
```

The pointer-based iterator types are contiguous random-access iterators and
can be passed directly to standard-library algorithms.

---

## Construction and Lifetime

### Default construction

```cpp
Buffer() noexcept;
```

Constructs an empty buffer with `size() == 0` and `data() == nullptr`.

Complexity: O(1).

### Size construction

```cpp
explicit Buffer(size_type size);
```

Constructs `size` elements from `value_type{}`. For scalar types, this yields
zero-initialized values. If element construction fails, constructed elements
and the allocation are cleaned up before the exception is propagated.

Complexity: O(size).

Throws:

- `std::bad_array_new_length` if `size > max_size()`
- `std::bad_alloc` if allocation fails
- Any exception thrown while constructing an element

### Fill construction

```cpp
Buffer(size_type size, const_reference value);
```

Copy-constructs `size` elements from `value`, with cleanup if a copy fails.

Complexity: O(size).

Throws the allocation exceptions above or an exception from `T`'s copy
constructor.

### Initializer-list construction

```cpp
Buffer(std::initializer_list<value_type> list);
```

Copy-constructs elements in list order.

Complexity: O(list.size()).

Throws the allocation exceptions above or an exception from `T`'s copy
constructor.

### Copy construction

```cpp
Buffer(const Buffer& other);
```

Creates an independent allocation containing copies of `other`'s elements.
Partially constructed state is cleaned up if copying fails.

Complexity: O(other.size()).

### Move construction

```cpp
Buffer(Buffer&& other) noexcept;
```

Transfers the allocation without moving individual elements. `other` becomes
empty. Existing pointers and iterators into `other` continue to refer to the
same elements, now owned by the destination.

Complexity: O(1).

### Destruction

```cpp
~Buffer();
```

Destroys every element and releases the aligned allocation.

Complexity: O(size()).

---

## Assignment

### Copy assignment

```cpp
Buffer& operator=(const Buffer& other);
```

Uses copy-and-swap to provide the strong exception guarantee. If allocation or
copying fails, the destination is unchanged. Successful assignment invalidates
references, pointers, and iterators into the destination's old storage.

Complexity: O(size() + other.size()).

### Move assignment

```cpp
Buffer& operator=(Buffer&& other) noexcept;
```

Destroys the destination's current elements, releases its allocation, and then
transfers ownership from `other`. The source becomes empty. Self-move
assignment has no effect.

Complexity: O(size()) for destruction of the old elements; ownership transfer
is O(1).

---

## Element Access

### Unchecked access

```cpp
reference operator[](size_type index) noexcept;
const_reference operator[](size_type index) const noexcept;
```

Returns the element at `index` without bounds checking. Behavior is undefined
unless `index < size()`.

Complexity: O(1).

### Front and back

```cpp
reference front();
const_reference front() const;
reference back();
const_reference back() const;
```

Returns the first or final element. Each function throws
`Exceptions::IndexError` when the buffer is empty.

Complexity: O(1).

### Data access

```cpp
[[nodiscard]] pointer data() noexcept;
[[nodiscard]] const_pointer data() const noexcept;
```

Returns the first element's address, or `nullptr` for an empty buffer. For a
non-empty buffer, the live element range is `[data(), data() + size())`.

Complexity: O(1).

---

## Iteration

`begin`, `end`, `cbegin`, and `cend` expose the contiguous forward range.
`rbegin`, `rend`, `crbegin`, and `crend` expose the same elements in reverse
order. Every iterator accessor is O(1).

For an empty buffer:

```cpp
buffer.begin()  == buffer.end();
buffer.cbegin() == buffer.cend();
buffer.rbegin() == buffer.rend();
```

Traversing the complete range is O(size()).

---

## Capacity and Allocation Metadata

```cpp
[[nodiscard]] static constexpr size_type alignment() noexcept;
[[nodiscard]] static constexpr size_type max_size() noexcept;
[[nodiscard]] size_type size() const noexcept;
[[nodiscard]] bool empty() const noexcept;
```

- `alignment()` returns the requested allocation alignment in bytes.
- `max_size()` returns the largest count whose byte size cannot overflow;
  allocation of that many elements is not guaranteed to succeed.
- `size()` returns the number of live elements.
- `empty()` is equivalent to `size() == 0`.

Each operation is O(1).

---

## Modifiers

### fill

```cpp
void fill(const_reference value);
```

Copy-assigns `value` to each element from first to last. If an assignment
throws, earlier elements retain their new values and later elements retain
their previous values.

Complexity: O(size()).

### swap

```cpp
void swap(Buffer& other) noexcept;
```

Exchanges the allocation and element count without moving or copying
individual elements. Pointers, references, and iterators remain valid but now
refer to elements owned by the other buffer.

Complexity: O(1).

---

## Complexity Summary

| Operation | Complexity |
| --------- | ---------- |
| Default construction | O(1) |
| Size/fill/list construction | O(n) |
| Copy construction | O(n) |
| Move construction | O(1) |
| Copy assignment | O(old size + copied size) |
| Move assignment | O(old size) |
| Destruction | O(n) |
| Individual access or iterator creation | O(1) |
| Full traversal | O(n) |
| `fill()` | O(n) |
| `swap()` | O(1) |

---

## Design Notes

The class deliberately has no capacity, resize, insertion, or erasure API.
This keeps allocation ownership simple and makes it suitable as the stable
storage layer beneath higher-level array metadata and containers.

Aligned allocation uses C++ aligned `operator new` and the matching aligned
`operator delete`. Element construction is separate from allocation so cleanup
can be performed correctly when a constructor throws.

---

## See Also

- @ref arraybase
- @ref shape
- @ref strides
- @ref vector
- @ref matrix
- @ref tensor
