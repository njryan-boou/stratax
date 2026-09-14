@page validation Validation

# Validation {#dev_validation}

## Shared numeric checks

`include/stratax/core/validation/NumericValidation.hpp` keeps reusable numeric
checks in `stratax::core::validation`. Arithmetic and bitwise operations use
these internal helpers before evaluating potentially invalid expressions:

| Helper | Contract |
| --- | --- |
| `require_valid_division(lhs, rhs)` | Reject zero divisors and native signed minimum divided by minus one |
| `valid_shift_count<Value>(count)` | Test whether count is nonnegative and below the stored value type's bit width |
| `require_valid_shift_count<Value>(count)` | Raise ValueError when the shift count is invalid |

Division checks use native C++ expression types after integer promotions and
usual arithmetic conversions. They do not check general arithmetic overflow or
conversion to the result dtype. Scalar checks still apply to empty arrays where
the operator contract requires them; array operands are checked where used.
Compound division and shifts validate all used operands before writing.
See @ref numerical_contract, @ref arithmetic, and @ref bitwise.

Checks specific to a container's state remain with that container. The older
DimensionValidation, IndexValidation, ShapeValidation, TypeValidation, and
Validation headers remain empty compatibility headers; new code should include
the concrete helper it uses.

## Index normalization and unchecked access

Checked signed index normalization maps an index to a valid position and is
provided by the indexing module:

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    assert(stratax::indexing::normalize_index(-1, 4) == 3);
}
```

It accepts positive and Python-style negative indices and throws
`Exceptions::IndexError` when the index is outside the valid range.

Unchecked accessors still have caller preconditions; not every invariant is
checked on every access. The reserved bounds_checking flag does not change this.
