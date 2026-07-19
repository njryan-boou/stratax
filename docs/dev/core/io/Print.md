# Print

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/io/Print.hpp`

---

## Overview

`Print.hpp` implements stream insertion operators for Stratax containers.

Vector and matrix printing route through tensor conversion/formatting helpers, and tensor output uses recursive nested bracket formatting.

---

## Responsibilities

The print module is responsible for:

- Providing `operator<<` overloads for `Vector`, `Matrix`, and `Tensor`
- Emitting stable human-readable nested bracket formatting
- Applying indentation and sibling separators for multidimensional output

The print module is not responsible for:

- Locale-specific number formatting policies
- Truncation/pretty-print options for very large outputs
- Validation or repair of container metadata

---

## Relationships

```text
operator<<(Vector)
└── to_tensor(vector) -> detail::print_tensor_like

operator<<(Matrix)
└── to_tensor(matrix) -> detail::print_matrix_like

operator<<(Tensor)
└── detail::print_tensor_like
    └── detail::print_tensor_recursive
```

Depends on:

- `include/stratax/core/containers/Vector.hpp`
- `include/stratax/core/containers/Matrix.hpp`
- `include/stratax/core/containers/Tensor.hpp`
- `include/stratax/core/algorithms/Conversions.hpp`

---

## Invariants

The following conditions are always true:

- Printing does not mutate input containers.
- Empty tensor-like outputs print as `[]`.
- Tensor recursion respects shape and strides.
- Indentation depth increases by 4 spaces per nesting level.
- Matrix path uses newline-separated sibling groups (no comma between rows).
- Tensor path uses comma-newline sibling separators.

---

## Public Interface

### Vector stream output

```cpp
template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& vector);
```

Behavior

- Converts vector to tensor representation via `to_tensor`
- Prints using tensor-like formatter

Complexity

- O(n) output work

### Matrix stream output

```cpp
template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix);
```

Behavior

- Converts matrix to tensor representation via `to_tensor`
- Prints via matrix-like separator mode

Complexity

- O(n) output work

### Tensor stream output

```cpp
template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor);
```

Behavior

- Prints empty tensors as `[]`
- Otherwise recursively prints nested dimensions

Complexity

- O(n + s), where `s` is emitted structural separator/indent text

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Vector output | O(n) |
| Matrix output | O(n) |
| Tensor output | O(n + s) |
| Empty tensor output | O(1) |

`n` is element count.

---

## Examples

```cpp
std::cout << vec << '\n';
std::cout << mat << '\n';
std::cout << tensor << '\n';
```

---

## Design Notes

Using shared recursive tensor formatting keeps output behavior consistent across container families while still allowing matrix-specific sibling separation.

Whitespace and separator stability is important because tests and docs often compare emitted strings.

---

## Future Improvements

- Add configurable formatting profiles (compact vs pretty)
- Add optional truncation/summarization for large arrays
- Add custom indentation and separator settings

---

## See Also

- `include/stratax/core/algorithms/Conversions.hpp`
- `include/stratax/core/containers/Tensor.hpp`
