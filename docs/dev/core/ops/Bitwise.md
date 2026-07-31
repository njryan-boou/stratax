@page bitwise Bitwise

# Bitwise {#dev_bitwise}

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/ops/Bitwise.hpp`

---

## Overview

`Bitwise.hpp` defines generic element-wise bitwise operations for Stratax array-like containers whose value type is integer.

It provides unary, array-array, array-scalar, scalar-array (reverse), and compound assignment bitwise operators.

---

## Responsibilities

The bitwise module is responsible for:

- Enforcing shape compatibility for array-array bitwise operations
- Producing element-wise bitwise results with preserved shape
- Restricting bitwise operators to integer array/scalar combinations via concept constraints
- Providing in-place compound assignment operators

The bitwise module is not responsible for:

- Broadcasting or automatic shape expansion
- Type-promotion policy beyond C++ operator semantics
- Validating shift-count ranges beyond core language/operator behavior

---

## Relationships

```text
Bitwise operators
├── Array concept constraints
├── Integer concept constraints
├── validation::require_same_shape(...)
└── in-place operators forwarding to non-compound operators
```

Depends on:

- `include/stratax/core/Concepts.hpp`
- `include/stratax/core/validation/Validation.hpp`

Used by:

- User-facing vector/matrix/tensor integer bitwise expressions

---

## Invariants

The following conditions are always true:

- Array-array operators require identical shape.
- Result shape matches the array operand shape.
- Non-compound operators do not mutate inputs.
- Compound operators are implemented via non-compound operators and assignment.
- Bitwise operators are available only for integer-valued arrays/scalars.

---

## Public Interface

### Shape guard

```cpp
template<Array A>
requires Integer<typename A::value_type>
void require_same_bitwise_shape(const A& lhs, const A& rhs);
```

Throws

- `Exceptions::ShapeError` on shape mismatch

Complexity

- O(r)

### Unary operator

```cpp
template<Array A>
requires Integer<typename A::value_type>
A operator~(const A& value);
```

Complexity

- O(n)

### Array-array operators

```cpp
template<Array A> requires Integer<typename A::value_type>
A operator&(const A& lhs, const A& rhs);

template<Array A> requires Integer<typename A::value_type>
A operator|(const A& lhs, const A& rhs);

template<Array A> requires Integer<typename A::value_type>
A operator^(const A& lhs, const A& rhs);

template<Array A> requires Integer<typename A::value_type>
A operator<<(const A& lhs, const A& rhs);

template<Array A> requires Integer<typename A::value_type>
A operator>>(const A& lhs, const A& rhs);
```

Throws

- `Exceptions::ShapeError` on shape mismatch

Complexity

- O(n), plus O(r) shape check

### Array-scalar operators

```cpp
template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator&(const A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator|(const A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator^(const A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator<<(const A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator>>(const A& lhs, const Scalar& rhs);
```

Complexity

- O(n)

### Scalar-array operators

```cpp
template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator&(const Scalar& lhs, const A& rhs);

template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator|(const Scalar& lhs, const A& rhs);

template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator^(const Scalar& lhs, const A& rhs);

template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator<<(const Scalar& lhs, const A& rhs);

template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator>>(const Scalar& lhs, const A& rhs);
```

Complexity

- O(n)

### Compound assignment operators

```cpp
template<Array A> requires Integer<typename A::value_type>
A& operator&=(A& lhs, const A& rhs);

template<Array A> requires Integer<typename A::value_type>
A& operator|=(A& lhs, const A& rhs);

template<Array A> requires Integer<typename A::value_type>
A& operator^=(A& lhs, const A& rhs);

template<Array A> requires Integer<typename A::value_type>
A& operator<<=(A& lhs, const A& rhs);

template<Array A> requires Integer<typename A::value_type>
A& operator>>=(A& lhs, const A& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator&=(A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator|=(A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator^=(A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator<<=(A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator>>=(A& lhs, const Scalar& rhs);
```

Throws

- Same categories as corresponding non-compound operator

Complexity

- O(n), plus O(r) shape check for array-array forms

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| `require_same_bitwise_shape` | O(r) |
| Unary `~` | O(n) |
| Array-array operators | O(n + r) |
| Array-scalar operators | O(n) |
| Scalar-array operators | O(n) |
| Compound assignment | O(n + r) for array-array, O(n) for array-scalar |

`n` is element count and `r` is rank.

---

## Examples

```cpp
const auto a_and_b = a & b;
const auto with_mask = a & 0x0F;
const auto toggled = 0xFF ^ a;

const auto left_shifted = a << 2;
const auto right_shifted = a >> 1;

a &= b;
a |= 0x80;
a <<= 1;
```

---

## Design Notes

Bitwise operators are constrained to integer-valued arrays to keep semantics explicit and avoid accidental use with floating-point or complex container types.

Shared helper functions implement array-array, array-scalar, and scalar-array traversal so the public operators remain small forwarding wrappers.

The AND, OR, XOR, and NOT operators use standard function objects where available; shift operators use local callables because the standard library does not provide equivalent shift function objects.

Compound assignments delegate to non-compound operators to centralize shape validation and operation behavior.

---

## Future Improvements

- Explicit policies for shift-count validation and diagnostics
- Optional unsigned/signed-specific bitwise utility helpers
- SIMD kernels for common integer element types

---

## See Also

- `include/stratax/core/Concepts.hpp`
- `include/stratax/core/ops/Arithmetic.hpp`
- `include/stratax/core/ops/Comparison.hpp`
