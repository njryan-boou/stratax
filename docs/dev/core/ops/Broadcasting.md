@page broadcasting Broadcasting

# Broadcasting {#dev_broadcasting}

Version: v0.2.0

Status: Complete

Header: `include/stratax/ops/Broadcasting.hpp`

---

## Overview

`Broadcasting.hpp` defines shape compatibility, result-shape calculation, and
element-wise traversal for operands with different but compatible shapes.

Shapes align from the trailing dimension. Two aligned dimensions are
compatible when they are equal or either dimension has length one. Missing
leading dimensions behave as dimensions of length one.

---

## Responsibilities

The broadcasting module is responsible for:

- Validating whether two shapes can be broadcast together
- Computing the common broadcasted result shape
- Projecting result coordinates into each operand
- Applying binary operations to array-array, array-scalar, and scalar-array operands
- Reporting incompatible shapes with `Exceptions::BroadcastError`

The broadcasting module is not responsible for:

- Choosing a specific arithmetic or logical operation
- Numeric type promotion
- Division-by-zero policy
- Mutating either input operand

---

## Relationships

```text
broadcasted_op(...)
|-- validate_broadcast(...)
|-- broadcasted_shape(...)
|-- trailing-dimension coordinate projection
`-- operation callable supplied by the caller
```

Depends on:

- `include/stratax/concepts/Numeric.hpp`
- `include/stratax/core/Shape.hpp`
- `include/stratax/exceptions/Exceptions.hpp`

Used by:

- `include/stratax/ops/Arithmetic.hpp`
- Direct custom element-wise operations

---

## Broadcasting Rules

Dimensions are compared from right to left:

```text
left:       (2, 3, 1)
right:         (1, 4)
result:     (2, 3, 4)
```

For each aligned pair:

- Equal dimensions retain their length.
- A dimension of length one expands to the other length.
- A missing leading dimension expands to the other length.
- Any other mismatch is incompatible.

During traversal, an expanded singleton dimension always reads coordinate zero
from its source operand.

---

## Invariants

The following conditions are always true:

- Broadcasting aligns shapes from the trailing dimension.
- Input arrays are never mutated.
- Array-array output uses the common broadcasted shape.
- Scalar operations preserve the array shape.
- Operand order is preserved when invoking the operation callable.
- Incompatible array shapes raise `Exceptions::BroadcastError`.

---

## Public Interface

### Shape validation

```cpp
bool validate_broadcast(
    const stratax::core::Shape& shape1,
    const stratax::core::Shape& shape2);
```

Returns `true` when every aligned dimension pair is compatible. This function
does not throw for ordinary incompatibility.

Complexity

- O(r)

### Result shape

```cpp
stratax::core::Shape broadcasted_shape(
    const stratax::core::Shape& shape1,
    const stratax::core::Shape& shape2);
```

Throws

- `Exceptions::BroadcastError` when the shapes are incompatible

Complexity

- O(r)

### Array-array operation

```cpp
template<Array A, typename Op>
A broadcasted_op(const A& lhs, const A& rhs, Op op);
```

Both arrays must have the same container and element type. The callable receives
the projected left and right values for each output coordinate.

Throws

- `Exceptions::BroadcastError` when the shapes are incompatible
- Any exception raised by `op`

Complexity

- O(nr)

### Array-scalar operation

```cpp
template<Array A, Numeric S, typename Op>
A broadcasted_op(const A& lhs, const S& rhs, Op op);
```

The callable receives each array value followed by the scalar. The result shape
matches `lhs`.

Complexity

- O(n)

### Scalar-array operation

```cpp
template<Numeric S, Array A, typename Op>
A broadcasted_op(const S& lhs, const A& rhs, Op op);
```

The callable receives the scalar followed by each array value. The result shape
matches `rhs`.

Complexity

- O(n)

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Shape validation | O(r) |
| Result-shape calculation | O(r) |
| Array-array operation | O(nr) |
| Array-scalar operation | O(n) |
| Scalar-array operation | O(n) |

`n` is result element count and `r` is result rank.

---

## Examples

```cpp
#include <functional>
#include <stratax.h>

stratax::Matrix<int> column{{1}, {2}};
stratax::Matrix<int> row{{10, 20, 30}};

const auto shape = stratax::broadcasted_shape(column.shape(), row.shape());
// shape is (2, 3)

const auto sum = stratax::broadcasted_op(column, row, std::plus<>{});
const auto shifted = stratax::broadcasted_op(sum, 5, std::plus<>{});
const auto reverse = stratax::broadcasted_op(100, shifted, std::minus<>{});
```

The functions are also available through the grouped
`stratax::broadcasting` namespace.

---

## Design Notes

The public shape operations remain separate from traversal so callers can
validate or inspect a result shape without allocating a result array.

Flat operand offsets are computed from output coordinates without allocating a
coordinate vector for each element.

---

## Future Improvements

- Support mixed container and element types with an explicit result policy
- Add optimized contiguous and repeated-block traversal paths
- Add optional parallel broadcasting kernels

---

## See Also

- @ref arithmetic "Arithmetic"
- `include/stratax/core/Shape.hpp`
- `include/stratax/exceptions/Exceptions.hpp`
