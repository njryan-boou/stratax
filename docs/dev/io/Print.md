# Print

Developer notes for `include/stratax/io/Print.hpp`.

## Purpose

Defines stream output formatting for Stratax containers.

## Main API

### Vector Output
- `operator<<(std::ostream&, const Vector<T>&)`

### Matrix Output
- `operator<<(std::ostream&, const Matrix<T>&)`

### Tensor Output
- `operator<<(std::ostream&, const Tensor<T>&)`

## Validation Notes

- Printing assumes container shape metadata is valid.
- Matrix printing uses `rows()`, `cols()`, and `operator()(row, col)`.
- Tensor printing uses recursive formatting with shape and strides.
- No formatting validation is performed.

## Implementation Notes

- Vector values are separated by commas and spaces.
- Matrix rows are separated by newlines.
- Tensor dimensions are recursively nested and indented.
- Empty tensors print as `[]`.
- Keep whitespace stable for tests and examples.

## Future Work

- Add configurable separators.
- Add pretty formatting for large containers.
