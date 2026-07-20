# Strides {#dev_strides}

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/containers/Strides.hpp`

---

## Overview

`Strides` stores row-major stride metadata for a `Shape`.

It maps logical multi-dimensional indices to flat storage offsets and is used as lightweight layout metadata by Stratax containers.

---

## Responsibilities

The `Strides` class is responsible for:

- Computing row-major strides from a `Shape`
- Storing stride values contiguously
- Providing checked and unchecked stride access
- Providing iteration over stride values
- Supporting comparison and swap

The `Strides` class is **not** responsible for:

- Owning array element data
- Performing numerical algorithms
- Shape validation beyond stride-index access
- Multidimensional indexing logic itself

---

## Relationships

```text
Strides
│
└── buffer_ : Buffer<std::size_t>
    Stored stride values
```

Depends on:

- Buffer
- Shape
- Validation helpers (`validation::checked_multiply`, `validation::require_index`)
- Exceptions (`Exceptions::DimensionError`, `Exceptions::IndexError`)

Used by:

- Vector
- Matrix
- Tensor

Related classes:

- Shape
- Buffer

---

## Internal Data

| Member | Description |
| ------- | ----------- |
| `Buffer<std::size_t> buffer_` | Contiguous storage of stride values |

---

## Invariants

The following conditions are always true:

- `rank()` equals `size()`.
- Empty shape produces empty strides.
- For non-empty row-major shapes, the last stride is `1`.
- `at(index)` validates bounds and throws on invalid access.
- `operator==` compares both rank and all stride values.

---

## Public Interface

## Constructors

### Default Constructor

```cpp
Strides() noexcept;
```

Constructs an empty stride vector.

Complexity

- O(1)

Throws

- None

---

### Shape Constructor

```cpp
explicit Strides(const Shape& shape);
```

Constructs row-major strides for a shape.

Complexity

- O(n)

Throws

- `std::bad_alloc`
- `Exceptions::DimensionError` on stride overflow

---

### Copy Constructor

```cpp
Strides(const Strides&) = default;
```

Complexity

- O(n)

---

### Move Constructor

```cpp
Strides(Strides&&) noexcept = default;
```

Complexity

- O(1)

---

### Destructor

```cpp
~Strides() = default;
```

Complexity

- O(n)

---

## Assignment Operators

## Copy Assignment

```cpp
Strides& operator=(const Strides&) = default;
```

Complexity

- O(n)

---

## Move Assignment

```cpp
Strides& operator=(Strides&&) noexcept = default;
```

Complexity

- O(1)

---

## Methods

## size()

```cpp
[[nodiscard]] std::size_t size() const noexcept;
```

Returns the number of stored stride values.

Complexity

- O(1)

---

## rank()

```cpp
[[nodiscard]] std::size_t rank() const noexcept;
```

Returns the represented dimensional rank.

Complexity

- O(1)

---

## empty()

```cpp
[[nodiscard]] bool empty() const noexcept;
```

Returns `true` if no strides are stored.

Complexity

- O(1)

---

## at()

```cpp
const std::size_t& at(std::size_t index) const;
```

Returns a stride value with bounds checking.

Complexity

- O(1)

Throws

- `Exceptions::IndexError` if out of bounds

---

## front()

```cpp
const std::size_t& front() const;
```

Returns first stride.

Preconditions

- Strides must not be empty.

Complexity

- O(1)

---

## back()

```cpp
const std::size_t& back() const;
```

Returns last stride.

Preconditions

- Strides must not be empty.

Complexity

- O(1)

---

## data()

```cpp
[[nodiscard]] const std::size_t* data() const noexcept;
```

Returns pointer to first stride value.

Complexity

- O(1)

---

## begin()/end()/cbegin()/cend()

```cpp
[[nodiscard]] const std::size_t* begin() noexcept;
[[nodiscard]] const std::size_t* begin() const noexcept;
[[nodiscard]] const std::size_t* cbegin() const noexcept;
[[nodiscard]] const std::size_t* end() noexcept;
[[nodiscard]] const std::size_t* end() const noexcept;
[[nodiscard]] const std::size_t* cend() const noexcept;
```

Provides iteration over stride values.

Complexity

- O(1)

---

## rbegin()/rend()/crbegin()/crend()

```cpp
[[nodiscard]] std::reverse_iterator<const std::size_t*> rbegin() noexcept;
[[nodiscard]] std::reverse_iterator<const std::size_t*> rbegin() const noexcept;
[[nodiscard]] std::reverse_iterator<const std::size_t*> crbegin() const noexcept;
[[nodiscard]] std::reverse_iterator<const std::size_t*> rend() noexcept;
[[nodiscard]] std::reverse_iterator<const std::size_t*> rend() const noexcept;
[[nodiscard]] std::reverse_iterator<const std::size_t*> crend() const noexcept;
```

Provides reverse iteration over stride values.

Complexity

- O(1)

---

## swap()

```cpp
void swap(Strides& other) noexcept;
```

Exchanges stride storage with another instance.

Complexity

- O(1)

---

## Operators

## operator()

```cpp
const std::size_t& operator()(std::size_t index) const;
```

Returns stride value without bounds checking.

Complexity

- O(1)

See Also

```cpp
const std::size_t& at(std::size_t index) const;
```

---

## operator== / operator!=

```cpp
[[nodiscard]] bool operator==(const Strides& other) const noexcept;
[[nodiscard]] bool operator!=(const Strides& other) const noexcept;
```

Compares stride vectors by rank and values.

Complexity

- O(n)

---

## stream operator

```cpp
std::ostream& operator<<(std::ostream& os, const Strides& stride);
```

Writes tuple-like stride representation, for example `(12, 4, 1)` and `(1,)` for rank-1.

Complexity

- O(n)

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Default construction | O(1) |
| Shape construction | O(n) |
| Copy construction | O(n) |
| Move construction | O(1) |
| Copy assignment | O(n) |
| Move assignment | O(1) |
| Destruction | O(n) |
| `size()` / `rank()` / `empty()` | O(1) |
| `operator()` / `at()` | O(1) |
| `front()` / `back()` | O(1) |
| `operator==` / `operator!=` | O(n) |
| Iteration | O(n) |
| `swap()` | O(1) |

---

## Examples

## Building Strides From a Shape

```cpp
Shape shape(2, 3, 4);
Strides strides(shape);

std::cout << strides << '\n'; // (12, 4, 1)
```

---

## Accessing Strides

```cpp
Strides strides(Shape(2, 3, 4));

auto first = strides(0);   // unchecked
auto safe = strides.at(1); // checked
```

---

## Design Notes

`Strides` stores only layout metadata and intentionally avoids array value ownership.

By separating shape and stride metadata from value storage, Stratax keeps indexing and layout logic explicit and reusable across `Vector`, `Matrix`, and `Tensor`.

---

## Future Improvements

- Optional column-major stride construction helper
- Utility helpers for broadcast-aware stride derivation
- Additional constexpr-friendly constructors

---

## See Also

- [Shape](Shape.md)
- [Buffer](Buffer.md)
- [Vector](Vector.md)
- [Matrix](Matrix.md)
- [Tensor](Tensor.md)
