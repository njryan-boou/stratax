# Comparison

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/ops/Comparison.hpp`

---

## Overview

`Comparison.hpp` defines exact equality and inequality operators for Stratax array-like containers.

Comparison is shape-aware and value-based: arrays are equal only when shape and all flat element values match.

---

## Responsibilities

The comparison module is responsible for:

- Fast shape gate before element comparison
- Exact element-wise equality in flat iteration order
- Defining inequality as the logical negation of equality

The comparison module is not responsible for:

- Approximate floating-point comparison
- Broadcasting or implicit shape alignment
- Ordering comparisons (`<`, `<=`, `>`, `>=`)

---

## Relationships

```text
operator==
├── validation::same_shape(lhs, rhs)
└── element-wise iterator comparison

operator!=
└── !(lhs == rhs)
```

Depends on:

- `include/stratax/core/Concepts.hpp`
- `include/stratax/core/validation/Validation.hpp`

Used by:

- Equality checks in tests and API call sites

---

## Invariants

The following conditions are always true:

- Comparisons do not mutate operands.
- Shape mismatch produces `false` from `operator==`.
- `operator!=` is exactly `!(operator==)`.
- Equality uses exact value comparison (`==`) per element.

---

## Public Interface

### Equality

```cpp
template<Array A>
[[nodiscard]] bool operator==(const A& lhs, const A& rhs);
```

Behavior

- Returns `false` immediately on shape mismatch
- Otherwise compares all elements in flat iteration order

Throws

- None expected from module logic for shape mismatch paths

Complexity

- Best case O(1) (early mismatch)
- Worst case O(r + n)

### Inequality

```cpp
template<Array A>
[[nodiscard]] bool operator!=(const A& lhs, const A& rhs);
```

Behavior

- Delegates to `operator==` and negates the result

Complexity

- Same as `operator==`

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Shape check | O(r) |
| Equality value scan | O(n) |
| `operator==` total | O(r + n), early exit possible |
| `operator!=` total | O(r + n), early exit possible |

`r` is rank and `n` is element count.

---

## Examples

```cpp
const bool equal = (a == b);
const bool different = (a != b);
```

---

## Design Notes

Keeping comparison exact and minimal avoids surprising semantics at the core layer.

Approximate comparisons for floating-point values should be introduced as separate explicit utilities rather than overloading `operator==` behavior.

---

## Future Improvements

- Add tolerance-based comparison helpers for floating-point arrays
- Add element-wise boolean comparison utilities

---

## See Also

- `include/stratax/core/ops/Arithmetic.hpp`
- `include/stratax/core/validation/ShapeValidation.hpp`
