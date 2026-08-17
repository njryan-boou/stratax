@page slice Slice

# Slice {#dev_slice}

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/Slice.hpp`

---

## Overview

`stratax::core::Slice` represents a raw, half-open, strided one-dimensional
index range.

It stores signed `start`, `stop`, and nonzero `step` values unchanged. Positive
steps select indices while they are less than `stop`; negative steps select
indices while they are greater than `stop`. Container-specific normalization
and clamping happen later in the slicing algorithms.

---

## Responsibilities

`Slice` is responsible for:

- Storing half-open interval metadata (`[start, stop)`) with stride
- Enforcing the non-zero-step invariant
- Computing selected element count (`size()`) for positive and negative strides
- Comparing raw slice descriptions

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

- `include/stratax/exceptions/Exceptions.hpp`

Used by:

- `include/stratax/indexing/Slicing.hpp`

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

- The stop bound is exclusive in the direction selected by `step()`.
- `step()` is never zero.
- `size()` returns `0` when interval direction and bounds select no indices.
- `empty()` is equivalent to `size() == 0`.
- Bounds remain raw until a slicing algorithm resolves them against an extent.

---

## Public Interface

## Type Aliases

```cpp
using size_type = std::size_t;
using difference_type = std::ptrdiff_t;
```

`difference_type` represents signed bounds and steps. `size_type` represents
the number of indices selected by the raw range.

---

## Constructor

```cpp
Slice(difference_type start, difference_type stop, difference_type step = 1);
```

Creates a strided half-open range and stores all three arguments unchanged.
Negative bounds are not interpreted relative to a container during
construction.

Throws

- `Exceptions::IndexError` if `step == 0`

Complexity

- O(1)

---

## Accessors

```cpp
[[nodiscard]] difference_type start() const noexcept;
[[nodiscard]] difference_type stop() const noexcept;
[[nodiscard]] difference_type step() const noexcept;
[[nodiscard]] size_type size() const noexcept;
[[nodiscard]] bool empty() const noexcept;
```

`start()`, `stop()`, and `step()` return the stored raw values. `size()` computes
the number of generated indices without iterating or resolving the bounds
against a container:

- A positive-step range is empty when `start() >= stop()`.
- A negative-step range is empty when `start() <= stop()`.
- Otherwise, the directed distance is divided by the absolute step and rounded
  upward.

The intermediate signed distance and rounding arithmetic must be representable
by `difference_type`.

Complexity

- `start()` / `stop()` / `step()` / `empty()`: O(1)
- `size()`: O(1)

---

## Comparisons

```cpp
[[nodiscard]] bool operator==(const Slice& other) const noexcept;
```

Two slices are equal when their stored `start`, `stop`, and `step` values all
match. Equality compares the raw representation, so two slices that happen to
generate the same indices can still compare unequal. In C++20, `operator!=` is
automatically rewritten from `operator==`.

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

`Slice` keeps only local interval semantics; extent normalization and clamping
happen in indexing-level slicing helpers where the container shape is
available.

Using signed fields enables Python-like negative-index flows at higher layers
without forcing unsigned conversion too early.

---

## Future Improvements

- Add convenience creators for common ranges (for example full-range helpers).
- Consider optional utilities for composing/intersecting slices.
- Make `size()` arithmetic safe across the full `std::ptrdiff_t` range.

---

## See Also

- @ref dev_ops_slice
