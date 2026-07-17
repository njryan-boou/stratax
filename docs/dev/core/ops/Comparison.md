# Comparison

Developer notes for `include/stratax/core/ops/Comparison.hpp`.

## Purpose

Implements generic equality and inequality for Stratax array containers.

## Main API

### Array Comparison
- `operator==`
- `operator!=`

## Invariants

- Equality never mutates either operand.
- Size or shape mismatch means arrays are not equal.
- Equal arrays must have the same flat element values in storage order.
- `operator!=` is always the logical negation of `operator==`.
- Comparison returns `false` for mismatched arrays instead of throwing.

## Validation Notes

- Size or shape mismatch is checked with `core::validation::same_shape()` and returns `false`.
- Comparison does not throw for mismatched size or shape.
- Element comparison assumes both containers expose flat iteration in logical order.

## Implementation Notes

- Equality checks size and shape first.
- Values are compared using iterators.
- Inequality delegates to equality.
- Works with vectors, matrices, and tensors.
- Shape compatibility comparison should stay routed through `Validation.hpp`.

## Time Complexity

- Best case for mismatched size is `O(1)`.
- Shape comparison is `O(r)`.
- Equality is `O(r + n)` in the worst case.
- Inequality has the same complexity as equality.

## Future Work

- Add approximate equality for floating-point containers.
- Add element-wise comparison arrays.
- Add tolerance controls.
- Add SIMD comparison paths.
