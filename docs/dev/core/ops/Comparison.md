@page comparison Comparison

# Comparison {#dev_comparison}

Header: `include/stratax/ops/Comparison.hpp`.

Global named functions equal, not_equal, less, less_equal, greater, and
greater_equal have equivalent ==, !=, <, <=, >, and >= operators. They accept
two owning arrays or an array and scalar in either order. Results are owning
Boolean arrays, not a single truth value.

## Invariants and dtype requirements

Array-array operands broadcast. Matching families retain their family; mixed
families return Tensor<bool>. Scalar comparisons retain the array family and
shape. Neither operand changes. Incompatible shapes and an empty operand
supplying a nonempty result raise Exceptions::BroadcastError. Allocation errors
propagate. Allocating results require owning container result traits.

Equality is exact using native scalar ==, not tolerance-based. Ordering requires
Ordered for both dtypes, excluding complex; Boolean values are allowed. Native
NaN and mixed-type comparison behavior applies. Do not treat a result mask as a
whole-array Boolean condition. The internal comparison_detail::array_equal
helper compares exact shape and values without broadcasting; it is not a public
whole-array reduction API. Public all/any are not implemented.

```cpp
#include <stratax.h>
#include <cassert>
#include <algorithm>

int main() {
    stratax::container::Matrix<int> column{{1}, {3}};
    stratax::container::Vector<int> row{2, 3, 4};
    auto mask = column < row;
    assert(mask.shape() == stratax::core::Shape({2, 3}));
    assert(mask(0, 0) && !mask(1, 0));
    const bool all_pairs_less = std::all_of(mask.begin(), mask.end(),
        [](bool value) { return value; });
    assert(!all_pairs_less);
}
```

Broadcasted operations cost O((n + 1)r); scalar operations cost O(n + r).
Results allocate O(n + r) storage. Python exports the six named functions and
comparison operators with BoolVector/BoolMatrix/BoolTensor results. See
@ref python_api and @ref broadcasting.
