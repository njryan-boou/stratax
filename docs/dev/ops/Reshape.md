# Reshape

Developer notes for `include/stratax/ops/reshape.hpp`.

## Purpose

Provides shape-changing helpers that preserve flat storage order.

## Main API

### Reshape
- `reshape(array, shape)`

### Flatten
- `flatten(array)`

## Validation Notes

- `reshape` requires `array.size() == shape.elements()`.
- Shape mismatch throws `Exceptions::ShapeError`.
- Overflow from `Shape::elements()` propagates as `Exceptions::DimensionError`.
- Empty and zero-dimension shapes are allowed when the element count matches.

## Implementation Notes

- `reshape` always returns a `Tensor<value_type>`.
- `flatten` always returns a rank-1 `Vector<value_type>`.
- Values are copied in flat storage order using `operator[]`.
- No view semantics are implemented; returned containers own their buffers.

## Future Work

- Add view-based reshape if non-owning containers are introduced.
- Add inferred dimension support, such as `{2, -1}`, only if signed shape syntax is added.
- Consider preserving container type where possible.
