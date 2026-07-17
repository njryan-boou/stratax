# Print

Developer notes for `include/stratax/core/io/Print.hpp`.

## Purpose

Defines stream output formatting for Stratax containers.

## Main API

### Vector Output
- `operator<<(std::ostream&, const Vector<T>&)`

### Matrix Output
- `operator<<(std::ostream&, const Matrix<T>&)`

### Tensor Output
- `operator<<(std::ostream&, const Tensor<T>&)`

## Invariants

- Printing never mutates the container.
- Output order follows flat row-major storage order within the displayed shape.
- Empty vectors and tensors print as `[]`.
- Matrix output is always bracketed across multiple lines.
- Non-empty tensor output uses nested brackets matching rank.
- Formatting is intentionally stable because tests assert exact strings.

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

## Time Complexity

- Vector printing is `O(n)`.
- Matrix printing is `O(n)`.
- Tensor printing is `O(n + b)`, where `b` is the number of nested bracket/indent separators emitted by the recursive formatter.
- Empty tensor printing is `O(r)` due to shape element-count checks.

## Future Work

- Add configurable separators.
- Add pretty formatting for large containers.
