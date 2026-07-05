# Comparison

Developer notes for `include/stratax/ops/Comparison.hpp`.

## Purpose

Implements generic equality and inequality for Stratax array containers.

## Main API

### Array Comparison
- `operator==`
- `operator!=`

## Validation Notes

- Shape mismatch returns `false`.
- Comparison does not throw for mismatched shapes.
- Element comparison assumes both containers expose flat iteration in logical order.

## Implementation Notes

- Equality checks shape first.
- Values are compared using iterators.
- Inequality delegates to equality.
- Works with vectors, matrices, and tensors.

## Future Work

- Add approximate equality for floating-point containers.
- Add element-wise comparison arrays.
- Add tolerance controls.
- Add SIMD comparison paths.
