# Shape

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/containers/Shape.hpp`

---

## Overview

`Shape` stores array dimension lengths and provides rank/element-count metadata for Stratax containers.

It is a lightweight metadata object built on `Buffer<std::size_t>` and is used by higher-level types such as `Strides`, `Vector`, `Matrix`, and `Tensor`.

---

## Responsibilities

The `Shape` class is responsible for:

- Storing dimension lengths
- Reporting rank
- Reporting whether a shape is empty
- Computing total element count with overflow checking
- Providing validated dimension access
- Supporting comparison and iteration over dimensions

The `Shape` class is **not** responsible for:

- Owning numeric array element storage
- Performing numerical algorithms
- Multidimensional data access into tensors/matrices/vectors
- Mathematical operations on data values

---

## Relationships

```text
Shape
│
└── dims_ : Buffer<std::size_t>
    Stored dimensions in axis order
```

Depends on:

- Buffer
- Validation helpers (`validation::checked_multiply`, `validation::require_index`, `validation::normalize_index`)
- Exceptions (`Exceptions::DimensionError`, `Exceptions::IndexError`)

Used by:

- Strides
- Vector
- Matrix
- Tensor

Related classes:

- Buffer
- Strides

---

## Internal Data

| Member | Description |
| ------- | ----------- |
| `Buffer<std::size_t> dims_` | Contiguous storage for all dimension lengths |

---

## Invariants

The following conditions are always true:

- `rank()` equals `dims_.size()`.
- `empty()` is equivalent to `rank() == 0`.
- Dimensions are stored in axis order from outermost to innermost.
- `operator==` compares both rank and all dimension values.
- `elements()` checks multiplication overflow and throws on overflow.

---

## Public Interface

## Constructors

### Default Constructor

```cpp
Shape() noexcept;
```

Constructs an empty rank-0 shape.

Complexity

- O(1)

Throws

- None

---

### Variadic Integral Constructor

```cpp
template<Integral... Dims>
requires (sizeof...(Dims) > 0)
Shape(Dims... dims);
```

Constructs a shape from one or more integral dimensions.

Complexity

- O(n)

Throws

- `std::bad_alloc`

---

### Initializer List + Tag Constructor

```cpp
Shape(std::initializer_list<std::size_t> list, allow_zero_t allow_zero);
```

Constructs a shape from explicit dimensions and documents zero-dimension intent.

Complexity

- O(n)

---

### Buffer Copy Constructor

```cpp
Shape(const Buffer<std::size_t>& dims);
```

Constructs a shape by copying dimensions from a buffer.

Complexity

- O(n)

---

### Buffer Copy + Tag Constructor

```cpp
Shape(const Buffer<std::size_t>& dims, allow_zero_t allow_zero);
```

Constructs a shape by copying dimensions from a buffer and documenting zero-dimension intent.

Complexity

- O(n)

---

### Buffer Move Constructor

```cpp
Shape(Buffer<std::size_t>&& dims);
```

Constructs a shape by taking ownership of dimension storage.

Complexity

- O(1)

---

### std::vector Constructor

```cpp
Shape(const std::vector<std::size_t>& dims);
```

Constructs a shape by copying dimensions from a standard vector.

Complexity

- O(n)

Throws

- `std::bad_alloc`

---

### Destructor

```cpp
~Shape() = default;
```

Default destructor.

Complexity

- O(n)

---

## Methods

## elements()

```cpp
[[nodiscard]] std::size_t elements() const;
```

Returns the product of all dimensions. Empty shape returns `0`.

Complexity

- O(n)

Throws

- `Exceptions::DimensionError` on overflow

---

## rank()

```cpp
[[nodiscard]] std::size_t rank() const;
```

Returns the number of stored dimensions.

Complexity

- O(1)

---

## empty()

```cpp
[[nodiscard]] bool empty() const noexcept;
```

Returns `true` if rank is zero.

Complexity

- O(1)

---

## swap()

```cpp
void swap(Shape& other) noexcept;
```

Exchanges dimension storage with another shape.

Complexity

- O(1)

---

## begin()/end()

```cpp
auto begin() noexcept;
auto end() noexcept;
auto begin() const noexcept;
auto end() const noexcept;
```

Provides mutable/const iteration over stored dimensions.

Complexity

- O(1)

---

## Assignment Operators

`Shape` uses compiler-generated copy and move assignment operators.

## Copy Assignment

```cpp
Shape& operator=(const Shape&) = default;
```

Complexity

- O(n)

---

## Move Assignment

```cpp
Shape& operator=(Shape&&) noexcept = default;
```

Complexity

- O(1)

---

## Operators

## operator()

```cpp
const std::size_t& operator()(std::size_t index) const;
```

Returns dimension at `index` with bounds validation.

Complexity

- O(1)

Throws

- `Exceptions::IndexError` if out of bounds

---

## operator[]

```cpp
const std::size_t& operator[](std::ptrdiff_t index) const;
```

Returns dimension at signed index. Negative indices count from the end.

Complexity

- O(1)

Throws

- `Exceptions::IndexError` if out of bounds

---

## operator== / operator!=

```cpp
[[nodiscard]] bool operator==(const Shape& other) const noexcept;
[[nodiscard]] bool operator!=(const Shape& other) const noexcept;
```

Compares two shapes by rank and all dimensions.

Complexity

- O(n)

---

## stream operator

```cpp
std::ostream& operator<<(std::ostream& os, const Shape& shape);
```

Writes shape in tuple-like form, for example `(2, 3, 4)` and `(5,)` for rank-1.

Complexity

- O(n)

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Default construction | O(1) |
| Variadic construction | O(n) |
| Initializer-list + tag construction | O(n) |
| Buffer copy construction | O(n) |
| Buffer move construction | O(1) |
| std::vector construction | O(n) |
| Copy assignment | O(n) |
| Move assignment | O(1) |
| Destruction | O(n) |
| `elements()` | O(n) |
| `rank()` | O(1) |
| `empty()` | O(1) |
| `operator()` | O(1) |
| `operator[]` | O(1) |
| `operator==` / `operator!=` | O(n) |
| Iteration | O(n) |
| `swap()` | O(1) |

---

## Examples

## Creating Shapes

```cpp
Shape a;
Shape b(2, 3, 4);
Shape c{std::vector<std::size_t>{1, 5, 7}};
```

---

## Accessing Dimensions

```cpp
Shape s(3, 224, 224);

auto first = s(0);
auto last = s[-1];
```

---

## Printing

```cpp
Shape s(3, 224, 224);
std::cout << s << '\n'; // (3, 224, 224)
```

---

## Design Notes

`Shape` stores only metadata and intentionally avoids owning numeric element data.

Dimension storage is contiguous through `Buffer<std::size_t>`, which keeps shape operations lightweight and iteration/cache friendly. Data containers (`Vector`, `Matrix`, `Tensor`) use `Shape` to define layout semantics.

---

## Future Improvements

- Add `cbegin()`/`cend()` convenience methods
- Add shape utility helpers (concat, slice, transpose-spec helpers)
- Add optional constexpr-friendly construction for fixed-rank usage

---

## See Also

- Buffer
- Strides
- Vector
- Matrix
- Tensor
