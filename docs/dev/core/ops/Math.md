@page math Math

# Math {#dev_math}

Header: `include/stratax/ops/Math.hpp`

---

## Overview

`Math.hpp` provides element-wise mathematical functions for Stratax arrays.
Unary functions preserve the input container and shape. Binary functions use
trailing-axis broadcasting and return the selected owning container and dtype.
These helpers require owning result traits; ArrayView has no supplied specialization.

Integral input to standard transcendental functions is promoted to
`dtype::float64`. Floating-point and complex inputs otherwise retain their
dtype. `abs()` returns the real component dtype for complex input.

---

## Unary Functions

| Category | Functions |
| -------- | --------- |
| Roots | `sqrt`, `cbrt` |
| Exponentials | `exp`, `exp2`, `expm1` |
| Logarithms | `log`, `log2`, `log10`, `log1p`, `logb` |
| Trigonometric | `sin`, `cos`, `tan` |
| Inverse trigonometric | `asin`, `acos`, `atan` |
| Hyperbolic | `sinh`, `cosh`, `tanh` |
| Inverse hyperbolic | `asinh`, `acosh`, `atanh` |
| Magnitude | `abs` |
| Special | `erf`, `erfc`, `tgamma`, `lgamma` |
| Rounding | `floor`, `ceil`, `trunc`, `round`, `nearbyint`, `rint` |

Every unary operation returns independent storage with the input shape and
container type. Rounding functions require real numeric elements. Other unary
functions accept numeric elements, including complex values where supported by
the corresponding standard-library operation.

Complexity: O(n + r), where `n` is the input element count and `r` is rank.

---

## Binary Functions

```text
pow(lhs, rhs);
atan2(lhs, rhs);
hypot(lhs, rhs);
fmod(lhs, rhs);
remainder(lhs, rhs);
copysign(lhs, rhs);
fmax(lhs, rhs);
fmin(lhs, rhs);
fdim(lhs, rhs);
nextafter(lhs, rhs);
```

Both operands are arrays. Their shapes are broadcast to a common result shape,
and their dtypes and container types are promoted. Integral-only operands
produce `dtype::float64` results.

Throws `Exceptions::BroadcastError` for incompatible shapes or an empty operand
supplying a nonempty result. Allocation and callable exceptions propagate.

Complexity: O((n + 1)r), where `n` is result size and `r` is result rank.

---

## Example

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    stratax::container::Matrix<double> values{{1.0}, {4.0}};
    stratax::container::Vector<double> powers{1.0, 0.5};

    const auto roots = stratax::core::sqrt(values);
    const auto result = stratax::core::pow(values, powers);

    assert(result.shape() == stratax::core::Shape({2, 2}));
    assert(roots(1, 0) == 2.0);
}
```

---

## Scalar behavior

Arguments reach the underlying std function in their original dtypes, then
convert to the result dtype. No Stratax domain checks or NaN-skipping rules are
added. cbrt, exp2, and log2 require real inputs because their std functions have
no complex overload. abs requires an unambiguous scalar overload; unsigned
types that do not promote to int are unsupported. Rounding preserves dtype,
and round resolves halfway cases away from zero. Binary functions promote the
dtype, then replace integral result dtypes with float64.

## Invariants

- Inputs are never mutated.
- Unary results preserve input shape and element order.
- Binary results use the common broadcasted shape.
- Empty result domains do not invoke the scalar operation.
- An empty operand is rejected when broadcasting would require its values.

---

## See Also

- @ref broadcasting "Broadcasting"
- @ref arithmetic "Arithmetic"
- @ref concepts "DType Concepts"
