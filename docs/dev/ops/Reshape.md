# Reshape

Developer notes for `include/stratax/ops/reshape.hpp`.

## Purpose

Provides shape-changing helpers that preserve flat storage order.

## Main API

### Reshape
- `reshape(array, shape)`

### Flatten
- `flatten(array)`

## Invariants

- `reshape` preserves flat element order.
- `reshape` does not change element count.
- `reshape` always returns an owning `Tensor`.
- `flatten` always returns an owning rank-1 `Vector`.
- `flatten(array).size() == array.size()`.
- No reshape or flatten helper returns a view.

## Validation Notes

- `reshape` requires `array.size() == shape.elements()` through `core::validation::require_equal_size()`.
- Shape mismatch throws `Exceptions::ShapeError`.
- Overflow from `Shape::elements()` propagates as `Exceptions::DimensionError`.
- Empty and zero-dimension shapes are allowed when the element count matches.

## Implementation Notes

- `reshape` always returns a `Tensor<value_type>`.
- `flatten` always returns a rank-1 `Vector<value_type>`.
- Values are copied in flat storage order using `operator[]`.
- No view semantics are implemented; returned containers own their buffers.
- Element-count validation should stay routed through `Validation.hpp`.

## Time Complexity

- `reshape(array, shape)` is `O(n + r)`.
- `flatten(array)` is `O(n)`.
- Shape element-count validation is `O(r)`.

## Future Work

- Add view-based reshape if non-owning containers are introduced.
- Add inferred dimension support, such as `{2, -1}`, only if signed shape syntax is added.
- Consider preserving container type where possible.
