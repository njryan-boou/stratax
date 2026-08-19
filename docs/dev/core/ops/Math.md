@page math Math

# Math {#dev_math}

Version: v0.2.0

Status: Complete

Header: `include/stratax/ops/Math.hpp`

---

## Overview

`Math.hpp` provides element-wise mathematical functions for Stratax arrays.
Unary functions preserve the input container and shape. Binary functions use
NumPy-style broadcasting and return the promoted container and dtype.

Integral input to standard transcendental functions is promoted to
`dtype::float64`. Floating-point and complex inputs otherwise retain their
dtype. `abs()` returns the real component dtype for complex input.

---

## Unary Functions

| Category | Functions |
| -------- | --------- |
| Roots | `sqrt`, `cbrt` |
| Exponentials | `exp`, `exp2` |
| Logarithms | `log`, `log2`, `log10` |
| Trigonometric | `sin`, `cos`, `tan` |
| Inverse trigonometric | `asin`, `acos`, `atan` |
| Hyperbolic | `sinh`, `cosh`, `tanh` |
| Inverse hyperbolic | `asinh`, `acosh`, `atanh` |
| Magnitude | `abs` |
| Rounding | `floor`, `ceil`, `trunc`, `round` |

Every unary operation returns independent storage with the input shape and
container type. Rounding functions require real numeric elements. Other unary
functions accept numeric elements, including complex values where supported by
the corresponding standard-library operation.

Complexity: O(n), where `n` is the input element count.

---

## Binary Functions

```cpp
pow(lhs, rhs);
atan2(lhs, rhs);
hypot(lhs, rhs);
fmod(lhs, rhs);
remainder(lhs, rhs);
copysign(lhs, rhs);
fmax(lhs, rhs);
fmin(lhs, rhs);
```

Both operands are arrays. Their shapes are broadcast to a common result shape,
and their dtypes and container types are promoted. Integral-only operands
produce `dtype::float64` results.

Throws `Exceptions::BroadcastError` when the operand shapes are incompatible.

Complexity: O(nr), where `n` is result size and `r` is result rank.

---

## Example

```cpp
stratax::Matrix<double> values{{1.0}, {4.0}};
stratax::Vector<double> powers{1.0, 0.5};

const auto roots = stratax::core::sqrt(values);
const auto result = stratax::core::pow(values, powers);

// result shape is {2, 2} through broadcasting.
```

---

## Invariants

- Inputs are never mutated.
- Unary results preserve input shape and element order.
- Binary results use the common broadcasted shape.
- Empty inputs produce empty results without invoking the scalar operation.

---

## See Also

- @ref broadcasting "Broadcasting"
- @ref arithmetic "Arithmetic"
- @ref concepts "DType Concepts"

