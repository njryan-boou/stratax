# Validation {#dev_validation}

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/validation/Validation.hpp`

---

## Overview

`Validation.hpp` is the aggregation entry point for Stratax runtime validation helpers.

It re-exports shape, dimension, index, and type checks implemented in dedicated headers under `include/stratax/core/validation/`.

---

## Responsibilities

The validation module is responsible for:

- Providing small reusable checks for ranks, dimensions, and indices
- Preventing unsigned overflow in size and offset arithmetic
- Enforcing shape and type compatibility invariants
- Standardizing exception categories for validation failures

The validation module is **not** responsible for:

- Container storage or ownership
- Algorithmic dispatch and broadcasting policy
- Error logging/transport layers

---

## Relationships

```text
Validation.hpp
├── DimensionValidation.hpp
├── IndexValidation.hpp
├── ShapeValidation.hpp
└── TypeValidation.hpp
```

Depends on:

- `include/stratax/core/Exceptions.hpp`
- `include/stratax/core/Concepts.hpp` (type/category checks)

Used by:

- Containers (`Shape`, `Strides`, `Vector`, `Matrix`, `Tensor`)
- Ops (`Arithmetic`, `Indexing`, `Slice`)
- Algorithms and bindings

---

## Invariants

The following conditions are always true:

- Helpers do not mutate caller-owned state.
- Failure categories map to specific exception types:
  - dimension/rank/overflow -> `Exceptions::DimensionError`
  - shape compatibility -> `Exceptions::ShapeError`
  - index normalization/bounds -> `Exceptions::IndexError`
  - type mismatch -> `Exceptions::TypeError`
- `checked_multiply()` and `checked_add()` never silently wrap.
- `require_rank(object, ...)` returns the original reference on success.

---

## Public Interface

## Aggregation Header

```cpp
#include <stratax/core/validation/Validation.hpp>
```

This file is intentionally an umbrella include with no additional function definitions.

---

## Re-exported Helpers

### Dimension helpers

```cpp
std::size_t nonnegative_size(std::ptrdiff_t value, const char* message);
void require_rank(std::size_t actual, std::size_t expected, const char* message);

template<typename Ranked>
const Ranked& require_rank(const Ranked& object, std::size_t expected, const char* message);

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, const char* message);
std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char* message);
```

### Index helpers

```cpp
std::size_t nonnegative_index(std::ptrdiff_t value, const char* message);
std::size_t normalize_index(std::ptrdiff_t value, std::size_t size, const char* message);
void require_index(std::size_t index, std::size_t size, const char* message);
void require_at_most(std::size_t value, std::size_t upper, const char* message);
```

### Shape helpers

```cpp
std::size_t nonnegative_shape_dimension(std::ptrdiff_t value, const char* message);
std::size_t positive_shape_dimension(std::ptrdiff_t value, const char* message);
void require_positive_shape_dimension(std::size_t value, const char* message);

template<typename Lhs, typename Rhs>
bool same_shape(const Lhs& lhs, const Rhs& rhs);

template<typename Lhs, typename Rhs>
void require_same_shape(const Lhs& lhs, const Rhs& rhs, const char* message);

void require_equal_size(std::size_t lhs, std::size_t rhs, const char* message);
```

### Type helpers

```cpp
template<typename Actual, typename Expected>
void require_type(const char* message);

template<typename T>
void require_numeric_type(const char* message);

template<typename Lhs, typename Rhs>
void require_same_value_type(const Lhs& lhs, const Rhs& rhs, const char* message);
```

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Scalar/rank/index/type checks | O(1) |
| `checked_multiply` / `checked_add` | O(1) |
| `same_shape` / `require_same_shape` | O(r) |
| `require_same_value_type` | O(1) |

`r` is rank (number of dimensions).

---

## Examples

```cpp
const auto i = stratax::core::validation::normalize_index(
    user_index,
    vec.size(),
    "Vector index out of bounds.");

stratax::core::validation::require_same_shape(
    lhs,
    rhs,
    "Operands must have the same shape.");
```

---

## Design Notes

Validation helpers are header-only and allocation-free so callers can compose checks without introducing runtime infrastructure dependencies.

Centralizing these checks keeps error category behavior consistent across containers, ops, and algorithms.

---

## Future Improvements

- Add broadcast-compatibility validation helpers once broadcasting rules are finalized.
- Add optional diagnostic helpers that build richer messages in debug builds.

---

## See Also

- `include/stratax/core/validation/DimensionValidation.hpp`
- `include/stratax/core/validation/IndexValidation.hpp`
- `include/stratax/core/validation/ShapeValidation.hpp`
- `include/stratax/core/validation/TypeValidation.hpp`
