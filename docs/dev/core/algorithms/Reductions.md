@page reductions Reductions

# Reductions {#dev_reductions}

Header: `include/stratax/algorithms/Reductions.hpp`.
Functions are declared in the global `reduction` namespace.

## Global results

| Function | C++ result | Empty input |
| --- | --- | --- |
| `sum(arr)` | Input dtype through reduction_sum_t | Zero |
| `prod(arr)` | Input dtype through reduction_prod_t | One |
| `min(arr)`, `max(arr)` | Input dtype | IndexError |
| `argmin(arr)`, `argmax(arr)` | dtype::int64 | IndexError |
| `mean(arr)`, `var(arr)`, `std(arr)` | double | ZeroDivisionError |

Sum/product accept Numeric dtypes and do not automatically widen integers.
Extrema require Ordered dtypes, so complex ordering is unavailable.
Mean/variance/std require Numeric and Ordered (real numeric, excluding bool).
The exceptions above are in `Exceptions` in C++.

Global extrema indices are logical flat positions, with the first occurrence
winning ties under the scalar comparisons. There is no NaN-skipping policy.
Mean converts each input to long double before accumulation and division, then
returns double. It avoids the source-dtype integer accumulation used by sum.
Long-double precision is platform dependent; rounding, floating overflow, and
cancellation remain possible. Variance uses Welford's online recurrence with
double intermediates and divides by n; std is its square root. These are
population statistics without ddof. See @ref numerical_contract for native
promotion, nonfinite values, and accumulation limits.

## Axis results and invariants

All functions accept `(arr, axis)` and `(arr, axis, keepdims)`. Axis must be in
`[-rank, rank)`; invalid axes raise AxisError. Results always own Tensor storage.
Result dtypes match the global scalar types. Axis extrema indices are positions
along the reduced axis, not flat positions in the original array.

With keepdims true the axis has extent one; otherwise it is removed. Reducing
rank one without keepdims still returns shape `{1}` because rank-zero Tensor
has no scalar storage. An empty output domain invokes no slice callbacks. If
output positions exist but each reduction slice is empty, the global empty
rules above apply to each output position.

## Generic callback and ownership

`axis_reduce(array, axis, func, keepdims=false)` first copies the input to an
owning Tensor. Callback result type is inferred using a const Tensor argument.
Actual calls receive that Tensor for the rank-one scalar-result case, otherwise
strided ArrayView slices. A generic callable must handle both argument types
and return a supported scalar dtype. It must not retain references or views
into the temporary copy. Input storage is not modified; callback, conversion,
slicing, and allocation exceptions propagate.

## Example

```cpp
#include <stratax.h>
#include <cassert>
#include <type_traits>

int main() {
    stratax::container::Matrix<int> m{{3, 1, 4}, {2, 7, 1}};
    const auto sums = reduction::sum(m, -1, true);
    assert(sums.shape() == stratax::core::Shape({2, 1}));
    assert(sums[0] == 8 && sums[1] == 10);
    const auto indices = reduction::argmax(m, 1);
    static_assert(std::is_same_v<typename decltype(indices)::value_type, stratax::dtype::int64>);
    assert(indices[0] == 2 && indices[1] == 1);
    auto spans = reduction::axis_reduce(m, 1, [](const auto& part) {
        return reduction::max(part) - reduction::min(part);
    });
    assert(spans[0] == 3 && spans[1] == 6);
    stratax::container::Tensor<int> empty(stratax::core::Shape{0, 3});
    assert(reduction::sum(empty, 0)[0] == 0);
    assert(reduction::max(empty, 1).empty());
}
```

## Cost

For n input elements, rank r, and p output elements, global owning reductions
are O(n); views take O((n + 1)r). Built-in axis reductions take
O((n + p + 1)r), with O(n + p + r) auxiliary storage. Generic axis_reduce adds
callback work. Empty inputs can still produce nonempty axis output.
Python binds reductions for owning double arrays and uses IndexTensor for axis
argmin/argmax; see @ref python_api.
