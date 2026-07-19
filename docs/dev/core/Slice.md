# Slice

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/Slice.hpp`

---

## Overview

`stratax::core::Slice` represents a half-open strided one-dimensional index range.

It stores signed `start`, `stop`, and non-zero `step` values and is used by slicing ops for vectors, matrices, and tensors.

---

## Responsibilities

`Slice` is responsible for:

- Storing half-open interval metadata (`[start, stop)`) with stride
- Enforcing the non-zero-step invariant
- Computing selected element count (`size()`) for positive and negative strides

`Slice` is **not** responsible for:

- Validating bounds against a concrete container extent
- Copying or materializing sliced outputs
- Multidimensional slicing policy

---

## Relationships

```text
stratax::core::Slice
├── start_ : std::ptrdiff_t
├── stop_  : std::ptrdiff_t
└── step_  : std::ptrdiff_t
```

Depends on:

- `include/stratax/core/Exceptions.hpp`

Used by:

- `include/stratax/core/ops/Slice.hpp`

---

## Internal Data

| Member | Description |
| ------- | ----------- |
| `start_` | Inclusive start index |
| `stop_` | Exclusive stop index |
| `step_` | Non-zero stride |

---

## Invariants

The following conditions are always true:

- Interval semantics are half-open: `[start(), stop())`.
- `step()` is never zero.
- `size()` returns `0` when interval direction and bounds select no indices.
- `empty()` is equivalent to `size() == 0`.

---

## Public Interface

## Constructor

```cpp
Slice(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step = 1);
```

Creates a strided half-open range.

Throws

- `Exceptions::IndexError` if `step == 0`

Complexity

- O(1)

---

## Accessors

```cpp
[[nodiscard]] std::ptrdiff_t start() const noexcept;
[[nodiscard]] std::ptrdiff_t stop() const noexcept;
[[nodiscard]] std::ptrdiff_t step() const noexcept;
[[nodiscard]] std::size_t size() const noexcept;
[[nodiscard]] bool empty() const noexcept;
```

Returns interval metadata and computed selection size.

Complexity

- `start()` / `stop()` / `step()` / `empty()`: O(1)
- `size()`: O(1)

---

## Comparisons

```cpp
[[nodiscard]] bool operator==(const Slice& other) const noexcept;
[[nodiscard]] bool operator!=(const Slice& other) const noexcept;
```

Two slices are equal when `start`, `stop`, and `step` all match.

Complexity

- O(1)

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Construction | O(1) |
| `start()` / `stop()` / `step()` | O(1) |
| `size()` / `empty()` | O(1) |
| `operator==` / `operator!=` | O(1) |

---

## Examples

```cpp
const stratax::core::Slice every_other(0, 10, 2);  // 0,2,4,6,8
const stratax::core::Slice reverse(8, -1, -2);     // 8,6,4,2,0

if (!every_other.empty())
{
    const auto count = every_other.size();
    (void)count;
}
```

---

## Design Notes

`Slice` keeps only local interval semantics; extent normalization and clamping happen in ops-level slicing helpers where container shape is available.

Using signed fields enables Python-like negative-index flows at higher layers without forcing unsigned conversion too early.

---

## Future Improvements

- Add convenience creators for common ranges (for example full-range helpers).
- Consider optional utilities for composing/intersecting slices.

---

## See Also

- `include/stratax/core/ops/Slice.hpp`
