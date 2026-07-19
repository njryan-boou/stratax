# Buffer\<T\>

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/containers/Buffer.hpp`

---

## Overview

`Buffer<T>` is Stratax's low-level fixed-size contiguous storage container.

It owns dynamically allocated memory, provides RAII semantics, and supplies iterator support for all higher-level Stratax containers. `Buffer<T>` serves as the foundation upon which `Shape`, `Strides`, `Vector`, `Matrix`, and `Tensor` are built.

---

## Responsibilities

The `Buffer` class is responsible for:

- Allocating contiguous memory
- Managing memory ownership (RAII)
- Providing iterator support
- Providing element access
- Copy and move semantics
- Serving as the storage backend for Stratax containers

The `Buffer` class is **not** responsible for:

- Numerical algorithms
- Shape validation
- Bounds validation
- Mathematical operations
- Multidimensional indexing

---

## Relationships

```text
Buffer<T>
│
├── data_
│      Pointer to contiguous storage
│
└── size_
       Number of stored elements
```

Depends on:

- C++ Standard Library
- Memory utilities
- Iterator facilities

Used by:

- Shape
- Strides
- Vector
- Matrix
- Tensor

Related classes:

- Shape
- Strides

---

## Internal Data

| Member | Description |
| ------- | ----------- |
| `T* data_` | Pointer to contiguous memory owned by the buffer |
| `std::size_t size_` | Number of stored elements |

---

## Invariants

The following conditions are always true:

- `size()` is never negative.
- `data()` is either `nullptr` or points to `size()` contiguous elements.
- `Buffer` always owns the memory it manages.
- Copy operations perform deep copies.
- Move operations transfer ownership.
- Memory is released automatically when the buffer is destroyed.

---

## Public Interface

## Constructors

### Default Constructor

```cpp
Buffer();
```

Constructs an empty buffer that owns no storage.

Complexity

- O(1)

Throws

- None

---

### Size Constructor

```cpp
Buffer(std::size_t size);
```

Constructs a buffer containing `size` default-initialized elements.

Complexity

- O(n)

Throws

- `std::bad_alloc`

---

### Fill Constructor

```cpp
Buffer(std::size_t size, const T& value);
```

Constructs a buffer containing `size` copies of `value`.

Complexity

- O(n)

Throws

- `std::bad_alloc`

---

### Initializer List Constructor

```cpp
Buffer(std::initializer_list<T> list);
```

Constructs a buffer from an initializer list.

Complexity

- O(n)

Throws

- `std::bad_alloc`

---

### Copy Constructor

```cpp
Buffer(const Buffer& other);
```

Constructs a deep copy of another buffer.

Complexity

- O(n)

---

### Move Constructor

```cpp
Buffer(Buffer&& other) noexcept;
```

Transfers ownership from another buffer.

Complexity

- O(1)

---

### Destructor

```cpp
~Buffer();
```

Releases owned memory.

Complexity

- O(n)

---

## Assignment Operators

## Copy Assignment

```cpp
Buffer& operator=(const Buffer& other);
```

Performs a deep copy.

Complexity

- O(n)

---

## Move Assignment

```cpp
Buffer& operator=(Buffer&& other) noexcept;
```

Transfers ownership.

Complexity

- O(1)

---

## Methods

## front()

```cpp
T& front();
```

Returns the first element.

Preconditions

- Buffer must not be empty.

Complexity

- O(1)

See Also

```cpp
const T& front() const;
```

---

## back()

```cpp
T& back();
```

Returns the last element.

Preconditions

- Buffer must not be empty.

Complexity

- O(1)

See Also

```cpp
const T& back() const;
```

---

## data()

```cpp
[[nodiscard]] T* data() noexcept;
```

Returns a pointer to the first stored element.

Complexity

- O(1)

See Also

```cpp
[[nodiscard]] const T* data() const noexcept;
```

---

## size()

```cpp
[[nodiscard]] std::size_t size() const noexcept;
```

Returns the number of stored elements.

Complexity

- O(1)

---

## empty()

```cpp
[[nodiscard]] bool empty() const noexcept;
```

Returns `true` if the buffer contains no elements.

Complexity

- O(1)

---

## fill()

```cpp
void fill(const T& value);
```

Assigns every stored element the value `value`.

Preconditions

- Buffer has been constructed.

Postconditions

- Every stored element equals `value`.

Complexity

- O(n)

---

## swap()

```cpp
void swap(Buffer& other) noexcept;
```

Exchanges the contents of two buffers.

Complexity

- O(1)

---

## Operators

## operator[]

```cpp
T& operator[](std::size_t index) noexcept;
```

Returns a reference to an element.

Bounds checking is **not** performed.

Complexity

- O(1)

See Also

```cpp
const T& operator[](std::size_t index) const noexcept;
```

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Default construction | O(1) |
| Size construction | O(n) |
| Fill construction | O(n) |
| Initializer list construction | O(n) |
| Copy construction | O(n) |
| Move construction | O(1) |
| Copy assignment | O(n) |
| Move assignment | O(1) |
| Destruction | O(n) |
| Element access | O(1) |
| Iteration | O(n) |
| fill() | O(n) |
| swap() | O(1) |

---

## Examples

## Creating a Buffer

```cpp
Buffer<double> a(5);
```

---

## Initializer List

```cpp
Buffer<double> b{1.0, 2.0, 3.0};
```

---

## Filling

```cpp
Buffer<double> values(10);

values.fill(3.14);
```

---

## Iteration

```cpp
for (const auto& value : values)
{
    std::cout << value << '\n';
}
```

---

## Design Notes

`Buffer<T>` stores data in a single contiguous memory block.

Higher-level containers such as `Matrix` and `Tensor` interpret this one-dimensional storage using `Shape` and `Strides`. This design improves cache locality, simplifies memory management, and allows algorithms to operate on a consistent underlying representation.

`Buffer<T>` intentionally contains no mathematical logic. Its sole responsibility is memory ownership and element storage.

---

## Future Improvements

- Custom allocators
- `std::pmr` allocator support
- Additional memory alignment strategies
- SIMD-aware allocation
- Optional uninitialized allocation optimizations
- Small-buffer optimization (if ever justified)

---

## See Also

- Shape
- Strides
- Vector
- Matrix
- Tensor
