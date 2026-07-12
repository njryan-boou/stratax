# Arithmetic

Developer notes for `include/stratax/ops/arithmetic.hpp`.

## Purpose

Implements generic element-wise arithmetic algorithms for all Stratax array containers.

## Main API

### Array <-> Array
- `operator+`
- `operator-`
- `operator*`
- `operator/`

### Array <-> Scalar
- `operator+`
- `operator-`
- `operator*`
- `operator/`

### Scalar <-> Array
- Reverse arithmetic operators.

### Compound Assignment
- `operator+=`
- `operator-=`
- `operator*=`
- `operator/=`

### Unary
- Unary `+`
- Unary `-`

## Invariants

- Arithmetic returns containers with the same shape as the array operand.
- Array-array arithmetic requires identical size and shape.
- Arithmetic operations do not mutate inputs except compound assignment operators.
- No arithmetic operator performs broadcasting.
- Division by a scalar checks zero before writing result elements.
- Element-wise division and scalar-array division may fill part of a temporary result before detecting a later zero divisor; the input arrays are not mutated unless compound assignment completes successfully.
- Result containers own their storage.

## Validation Notes

- Shapes must match before element-wise array operations through `core::validation::require_same_shape()`.
- Division checks for division by zero.
- Broadcasting is **not** implemented.
- Arithmetic does **not** change container shape or size.

## Implementation Notes

- Algorithms are generic and operate on any container satisfying the `Array` concept.
- Containers are expected to provide:
  - `shape()`
  - `size()`
  - `begin()`
  - `end()`
  - construction from `Shape`
- Array-array operators allocate a new result container.
- Compound assignment operators modify the left-hand side.
- Reverse operators reuse existing implementations when possible (`+` and `*`).
- Scalar overloads accept `Numeric` scalars, including complex numbers.
- Shape compatibility checks should stay routed through `Validation.hpp`.

## Time Complexity

- Array-array arithmetic is `O(n)`.
- Array-scalar and scalar-array arithmetic are `O(n)`.
- Compound assignment is `O(n)` because it computes a result and assigns it back.
- Unary `+` returns a copy and is `O(n)`.
- Unary `-` is `O(n)`.
- Shape checks are `O(r)` because shape equality compares dimensions.

## Future Work

- Broadcasting
- Type promotion (`int + double -> double`)
- SIMD optimization
- Parallel execution
