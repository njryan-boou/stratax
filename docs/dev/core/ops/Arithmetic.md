# Arithmetic

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/ops/Arithmetic.hpp`

---

## Overview

`Arithmetic.hpp` defines generic element-wise arithmetic for Stratax array-like containers.

It provides array-array, array-scalar, scalar-array, compound assignment, and unary operators for types satisfying the `Array` and `Numeric` concepts.

---

## Responsibilities

The arithmetic module is responsible for:

- Enforcing shape compatibility for array-array operations
- Producing element-wise arithmetic results with preserved shape
- Validating division-by-zero conditions where required
- Providing in-place compound assignment operators

The arithmetic module is not responsible for:

- Broadcasting or automatic shape expansion
- Type-promotion policy beyond C++ operator semantics
- SIMD or parallel execution policy

---

## Relationships

```text
Arithmetic operators
├── Array concept constraints
├── Numeric concept constraints
├── validation::require_same_shape(...)
└── Exceptions::ZeroDivisionError for division checks
```

Depends on:

- `include/stratax/core/Concepts.hpp`
- `include/stratax/core/validation/Validation.hpp`
- `include/stratax/core/Exceptions.hpp`

Used by:

- User-facing vector/matrix/tensor arithmetic expressions

---

## Invariants

The following conditions are always true:

- Array-array operators require identical shape.
- Result shape matches the array operand shape.
- Non-compound operators do not mutate inputs.
- Compound operators are implemented via non-compound operators and assignment.
- Division by zero raises `Exceptions::ZeroDivisionError`.

---

## Public Interface

### Shape guard

```cpp
template<Array A>
void require_same_arithmetic_shape(const A& lhs, const A& rhs);
```

Throws

- `Exceptions::ShapeError` on shape mismatch

Complexity

- O(r)

### Array-array operators

```cpp
template<Array A> A operator+(const A& lhs, const A& rhs);
template<Array A> A operator-(const A& lhs, const A& rhs);
template<Array A> A operator*(const A& lhs, const A& rhs);
template<Array A> A operator/(const A& lhs, const A& rhs);
```

Throws

- `Exceptions::ShapeError` on shape mismatch
- `Exceptions::ZeroDivisionError` for division when a divisor element is zero

Complexity

- O(n), plus O(r) shape check

### Array-scalar operators

```cpp
template<Array A, Numeric Scalar> A operator+(const A& lhs, const Scalar& rhs);
template<Array A, Numeric Scalar> A operator-(const A& lhs, const Scalar& rhs);
template<Array A, Numeric Scalar> A operator*(const A& lhs, const Scalar& rhs);
template<Array A, Numeric Scalar> A operator/(const A& lhs, const Scalar& rhs);
```

Throws

- `Exceptions::ZeroDivisionError` for scalar division by zero in `operator/`

Complexity

- O(n)

### Scalar-array operators

```cpp
template<Numeric Scalar, Array A> A operator+(const Scalar& lhs, const A& rhs);
template<Numeric Scalar, Array A> A operator-(const Scalar& lhs, const A& rhs);
template<Numeric Scalar, Array A> A operator*(const Scalar& lhs, const A& rhs);
template<Numeric Scalar, Array A> A operator/(const Scalar& lhs, const A& rhs);
```

Throws

- `Exceptions::ZeroDivisionError` for `operator/` when any array element is zero

Complexity

- O(n)

### Compound assignment operators

```cpp
template<Array A> A& operator+=(A& lhs, const A& rhs);
template<Array A> A& operator-=(A& lhs, const A& rhs);
template<Array A> A& operator*=(A& lhs, const A& rhs);
template<Array A> A& operator/=(A& lhs, const A& rhs);

template<Array A, Numeric Scalar> A& operator+=(A& lhs, const Scalar& rhs);
template<Array A, Numeric Scalar> A& operator-=(A& lhs, const Scalar& rhs);
template<Array A, Numeric Scalar> A& operator*=(A& lhs, const Scalar& rhs);
template<Array A, Numeric Scalar> A& operator/=(A& lhs, const Scalar& rhs);
```

Throws

- Same categories as corresponding non-compound operator

Complexity

- O(n), plus O(r) shape check for array-array forms

### Unary operators

```cpp
template<Array A> A operator-(const A& arr);
template<Array A> A operator+(const A& arr);
```

Behavior

- Unary minus multiplies by `-1`
- Unary plus returns a copy

Complexity

- O(n)

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| `require_same_arithmetic_shape` | O(r) |
| Array-array operators | O(n + r) |
| Array-scalar operators | O(n) |
| Scalar-array operators | O(n) |
| Compound assignment | O(n + r) for array-array, O(n) for array-scalar |
| Unary plus/minus | O(n) |

`n` is element count and `r` is rank.

---

## Examples

```cpp
const auto sum = a + b;
const auto shifted = a + 2.0;
const auto inv = 1.0 / a;

a += b;
a *= 3.0;
```

---

## Design Notes

Operators are intentionally generic and concept-constrained so they work uniformly across vector, matrix, and tensor containers.

Compound assignments delegate to their non-compound counterparts to centralize validation and arithmetic behavior.

---

## Future Improvements

- Broadcasting semantics
- Explicit type-promotion policy controls
- SIMD kernels for common numeric types
- Optional parallel backends

---

## See Also

- `include/stratax/core/Concepts.hpp`
- `include/stratax/core/ops/Comparison.hpp`
- `include/stratax/core/validation/ShapeValidation.hpp`
