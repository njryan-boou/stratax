@page arraybase ArrayBase

# ArrayBase {#dev_arraybase}

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/ArrayBase.hpp`

---

## Overview

`ArrayBase<T>` is the shared base class for Stratax owning containers.

It centralizes contiguous storage ownership and common container surface area for `Vector<T>`, `Matrix<T>`, and `Tensor<T>`.

---

## Responsibilities

The `ArrayBase` class is responsible for:

- Storing shared members: `buffer_`, `shape_`, and `strides_`
- Providing common metadata access (`size`, `empty`, `rank`, `shape`, `strides`)
- Providing shared contiguous access (`data`, `operator[]`)
- Providing shared iterator and reverse-iterator APIs
- Providing shared modifiers (`fill`, `swap`)

The `ArrayBase` class is **not** responsible for:

- Container-specific constructors and rank contracts
- Multidimensional indexing semantics
- Checked indexing policy for each concrete container
- Numerical algorithms or broadcasting behavior

---

## Relationships

```text
ArrayBase<T>
│
├── shape_   : core::Shape
├── strides_ : core::Strides
└── buffer_  : core::Buffer<T>

Derived by:
  - Vector<T>
  - Matrix<T>
  - Tensor<T>
```

Depends on:

- Buffer
- Shape
- Strides
- Exceptions (`Exceptions::IndexError` for empty front/back access)

Used by:

- `Vector<T>`
- `Matrix<T>`
- `Tensor<T>`

Related classes:

- Buffer
- Shape
- Strides

---

## Internal Data

| Member | Description |
| ------- | ----------- |
| `core::Buffer<T> buffer_` | Contiguous element storage |
| `core::Shape shape_` | Logical shape metadata |
| `core::Strides strides_` | Row-major stride metadata |

---

## Invariants

The following conditions are always true:

- `size()` reflects `buffer_.size()`.
- `rank()` reflects `shape_.rank()`.
- `shape()` and `strides()` expose the metadata backing the container.
- `front()` and `back()` throw on empty storage.
- `operator[]` and iterator traversal are unchecked (container-style semantics).

---

## Public Interface

### Common metadata and accessors

```cpp
std::size_t size() const noexcept;
bool empty() const noexcept;
std::size_t rank() const noexcept;
const Shape& shape() const noexcept;
const Strides& strides() const noexcept;
T* data() noexcept;
const T* data() const noexcept;
```

### Element and iterator surface

```cpp
T& front();
const T& front() const;
T& back();
const T& back() const;
T& operator[](std::size_t index) noexcept;
const T& operator[](std::size_t index) const noexcept;

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

### Modifiers

```cpp
void fill(const T& value);
void swap(ArrayBase& other) noexcept;
```

---

## Design Notes

`ArrayBase<T>` intentionally keeps policy minimal. Concrete containers own constructor contracts and indexing semantics while sharing storage/metadata plumbing and iteration behavior through inheritance.

---

## See Also

- @ref vector "Vector"
- @ref matrix "Matrix"
- @ref tensor "Tensor"
- @ref buffer "Buffer"
- @ref shape "Shape"
- @ref strides "Strides"