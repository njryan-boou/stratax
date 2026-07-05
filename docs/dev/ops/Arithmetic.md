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

## Validation Notes

- Shapes must match before element-wise array operations.
- Division checks for division by zero.
- Broadcasting is **not** implemented.
- Arithmetic does **not** change container shape or size.

## Implementation Notes

- Algorithms are generic and operate on any container satisfying the `NDarray` concept.
- Containers are expected to provide:
  - `shape()`
  - `size()`
  - `begin()`
  - `end()`
  - construction from `Shape`
- Array-array operators allocate a new result container.
- Compound assignment operators modify the left-hand side.
- Reverse operators reuse existing implementations when possible (`+` and `*`).

## Future Work

- Broadcasting
- Type promotion (`int + double -> double`)
- Complex number support
- SIMD optimization
- Parallel execution
