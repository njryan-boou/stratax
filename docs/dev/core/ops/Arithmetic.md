@page arithmetic Arithmetic

# Arithmetic {#dev_arithmetic}

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/ops/Arithmetic.hpp`

---

## Overview

`Arithmetic.hpp` defines generic element-wise arithmetic for Stratax
array-like containers.

It provides array-array, array-scalar, scalar-array, compound assignment, and
unary operators for types satisfying the `Array` and `Numeric` concepts.
Array-array operations use the broadcasting rules defined by
`Broadcasting.hpp`.

---

## Responsibilities

The arithmetic module is responsible for:

- Forwarding array-array operations through the broadcasting engine
- Providing array-scalar and scalar-array arithmetic
- Validating division-by-zero conditions where required
- Providing in-place compound assignment operators

The arithmetic module is not responsible for:

- Defining shape compatibility and index projection rules
- Type-promotion policy beyond C++ operator semantics
- SIMD or parallel execution policy

---

## Relationships

```text
Arithmetic operators
|-- Array and Numeric concept constraints
|-- broadcasted_op(...) for binary traversal
|-- Exceptions::BroadcastError for incompatible shapes
`-- Exceptions::ZeroDivisionError for division checks
```

Depends on:

- `include/stratax/core/Concepts.hpp`
- `include/stratax/core/ops/Broadcasting.hpp`
- `include/stratax/core/Exceptions.hpp`

Used by:

- User-facing vector, matrix, and tensor arithmetic expressions

---

## Invariants

The following conditions are always true:

- Array-array operators require broadcast-compatible shapes.
- Array-array result shape is the common broadcasted shape.
- Scalar operations preserve the array operand shape.
- Non-compound operators do not mutate inputs.
- Compound operators delegate to non-compound operators and assignment.
- Division by zero raises `Exceptions::ZeroDivisionError`.

---

## Public Interface

### Array-array operators

```cpp
template<Array A> A operator+(const A& lhs, const A& rhs);
template<Array A> A operator-(const A& lhs, const A& rhs);
template<Array A> A operator*(const A& lhs, const A& rhs);
template<Array A> A operator/(const A& lhs, const A& rhs);
```

Throws

- `Exceptions::BroadcastError` when the shapes are incompatible
- `Exceptions::ZeroDivisionError` for division when a broadcasted divisor is zero

Complexity

- O(nr), where `n` is the result element count and `r` is result rank

### Array-scalar operators

```cpp
template<Array A, Numeric Scalar> A operator+(const A& lhs, const Scalar& rhs);
template<Array A, Numeric Scalar> A operator-(const A& lhs, const Scalar& rhs);
template<Array A, Numeric Scalar> A operator*(const A& lhs, const Scalar& rhs);
template<Array A, Numeric Scalar> A operator/(const A& lhs, const Scalar& rhs);
```

Throws

- `Exceptions::ZeroDivisionError` for division by a zero scalar

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

- `Exceptions::ZeroDivisionError` for division when any array divisor is zero

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

Array-array compound operations assign the complete broadcasted result back to
the left operand. The left operand may therefore acquire the broadcasted shape.

Throws

- The same exceptions as the corresponding non-compound operator

### Unary operators

```cpp
template<Array A> A operator-(const A& arr);
template<Array A> A operator+(const A& arr);
```

Unary minus negates each value. Unary plus returns a copy.

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Array-array operators | O(nr) |
| Array-scalar operators | O(n) |
| Scalar-array operators | O(n) |
| Array-array compound assignment | O(nr) |
| Scalar compound assignment | O(n) |
| Unary plus/minus | O(n) |

`n` is result element count and `r` is result rank.

---

## Examples

```cpp
stratax::Matrix<int> column{{1}, {2}};
stratax::Matrix<int> row{{10, 20, 30}};

const auto sum = column + row; // shape (2, 3)
const auto shifted = sum + 2;
const auto inverse = 120 / shifted;

column += row; // column now has shape (2, 3)
```

---

## Design Notes

Operators are intentionally generic and concept-constrained so they work uniformly across vector, matrix, and tensor containers.

Scalar operand order is preserved for noncommutative operations such as
subtraction and division.

---

## Future Improvements

- Explicit type-promotion policy controls
- SIMD kernels for common numeric types
- Optional parallel backends

---

## See Also

- @ref broadcasting "Broadcasting"
- `include/stratax/core/Concepts.hpp`
- `include/stratax/core/ops/Comparison.hpp`
