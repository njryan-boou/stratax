@page arithmetic Arithmetic

# Arithmetic {#dev_arithmetic}

Header: `include/stratax/ops/Arithmetic.hpp`. Exact declarations and template
constraints are documented in the generated header reference.

## Operations and results

`+`, `-`, `*`, and `/` support two owning arrays or an array and numeric scalar
in either order. Array-array operands broadcast from the trailing axes.
Same-family results retain Vector, Matrix, or Tensor; mixed families return
Tensor. Scalar operations retain the array family and shape. Result elements
use `promote_t` from the operand dtypes, including scalar dtypes.

The callable receives original element types and its result is converted to the
output dtype afterward. Native C++ integer division and conversion rules apply. Signed division of
the expression type's minimum by minus one raises `Exceptions::OverflowError`;
other signed arithmetic intermediates must remain representable. Numeric excludes bool. Allocating operations need
RebindArray/PromoteArray specializations, supplied for owning containers.

## In-place invariants and failures

`+=`, `-=`, `*=`, and `/=` write into existing left storage, preserving its
shape and dtype. Broadcasting that would change the left shape raises
`Exceptions::BroadcastError`. Values are converted back to the left dtype.
Compound operations use both operands' values before mutation, including
shifted/transposed overlaps and broadcasting from the destination. A scalar
referencing a destination element is copied first. If multiple logical left
indices name the same physical element, the last logical row-major write wins.
For example, applying `+=` to overlapping views of `{1, 2, 3, 4}`, with left
positions 1–3 and right positions 0–2, produces `{1, 3, 5, 7}`.

Operations involving a view or another non-owning Array stage all results before
writing. Owning-owning and owning-scalar operations update directly because
owning storage has no repeated offsets. Compound division validates every used
pair for zero and signed division overflow before writing; either error leaves
the destination unchanged. A throwing custom callable can leave partial writes
on the direct owning path. Callables must not modify operands or their metadata.

Array division rejects a zero divisor only when that element is used. Scalar
zero divisors are rejected even when the array is empty. Rank-zero arrays are
empty and cannot broadcast values into a nonempty result. Non-compound
operations allocate independent storage and leave inputs unchanged.

Unary plus copies the array object (a view copy still aliases storage). Unary
minus returns an owning negated array; its implementation requires a dtype
that can be initialized from `-1`, so unsigned dtypes are unsupported.

## Example

```cpp
#include <stratax.h>
#include <cassert>
#include <type_traits>

int main() {
    stratax::container::Matrix<int> column{{1}, {2}};
    stratax::container::Matrix<int> row{{10, 20, 30}};
    auto result = column + row;
    assert(result.shape() == stratax::core::Shape({2, 3}));
    auto promoted = result + 0.5;
    static_assert(std::is_same_v<typename decltype(promoted)::value_type, double>);
    assert(promoted(0, 0) == 11.5);
    try {
        column += row;
        assert(false);
    } catch (const Exceptions::BroadcastError&) {}
    assert(column.shape() == stratax::core::Shape({2, 1}));
    result += row;
    assert(result(1, 2) == 62);
}
```

## Cost

Broadcasted owning operations take O((n + 1)r), for result size n and rank r.
Scalar and unary owning operations take O(n + r), including metadata allocation.
In-place array operations take O((n + 1)r); scalar assignment takes O(n) for
owning arrays and O((n + 1)r) for views. Owning results allocate O(n + r) storage. Compound operations involving
non-owning arrays allocate O(n) temporary element storage; owning compound
operations need no temporary element buffer.

See @ref broadcasting, @ref types, and @ref comparison.

Shared numeric checks live in `core/validation/NumericValidation.hpp`.
Result staging is shared execution code in `ops/detail/Compound.hpp`; it is
internal to the compound operators rather than a validation API.
