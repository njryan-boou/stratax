@page reshape Reshape and Flatten

# Reshape and Flatten {#dev_reshape}

Header: `include/stratax/algorithms/Reshape.hpp`.

`stratax::manipulation::reshape(arr, shape)` copies into Tensor with the requested shape.
`stratax::manipulation::flatten(arr)` copies into a rank-one Vector. Both preserve logical
row-major order and dtype, including for C++ strided view inputs.

## Invariants and failures

Output storage is independent. Reshape requires `shape.elements() == arr.size()`
and throws `Exceptions::ShapeError` on mismatch. Shape/stride overflow and
allocation failures propagate. Dimensions are explicit unsigned extents; there
is no inferred `-1` axis and no in-place metadata change.
Rank-zero output is valid only for an empty source. Flattening any empty input
produces shape `{0}`. Shape-changing views are a separate, unimplemented API.

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    stratax::container::Vector<int> source{0, 1, 2, 3, 4, 5};
    auto selected = stratax::indexing::slice(source, stratax::core::Slice{0, 6, 2});
    auto tensor = stratax::manipulation::reshape(selected, stratax::core::Shape{1, 3});
    assert(tensor(0, 2) == 4);
    auto flat = stratax::manipulation::flatten(tensor);
    flat[0] = 9;
    assert(source[0] == 0 && tensor[0] == 0);
}
```

For n elements, source rank r, and target rank q, reshape takes O(n + q) on
owning inputs and O((n + 1)r + q) on views. Flatten takes O(n) on owning inputs
and O((n + 1)r) on views. Both allocate storage for n elements and result metadata.
Python exposes reshape and flatten on owning containers, not on ArrayView.

See @ref conversions and @ref arrayview.
