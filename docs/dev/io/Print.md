# Print

Developer notes for `include/stratax/io/Print.hpp`.

## Purpose

Defines stream output formatting for Stratax containers.

## Main API

### Vector Output
- `operator<<(std::ostream&, const Vector<T>&)`

### Matrix Output
- `operator<<(std::ostream&, const Matrix<T>&)`

## Validation Notes

- Printing assumes container shape metadata is valid.
- Matrix printing uses `rows()`, `cols()`, and `operator()(row, col)`.
- No formatting validation is performed.

## Implementation Notes

- Vector values are separated by spaces.
- Matrix rows are separated by newlines.
- Keep whitespace stable for tests and examples.

## Future Work

- Add tensor printing.
- Add configurable separators.
- Add pretty formatting for large containers.
- Add stream tests for exact output.
