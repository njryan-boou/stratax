@page indexing Indexing

# Indexing {#dev_indexing}

Headers: `include/stratax/indexing/Indexing.hpp` and `Normalize.hpp`.

`stratax::indexing::offset(strides, indices)` computes the sum of each normalized
index times its element stride. It is unchecked: the index container must have
exactly one component per stride, all components must address valid storage,
and arithmetic must be representable. Strides may be canonical or describe a
view; the result is a physical element offset, not necessarily a logical flat
position. Cost is O(r).

`stratax::indexing::normalize_index(index, size)` accepts a signed index in
`[-size, size)` and returns its unsigned nonnegative equivalent. It handles
negative indices without overflowing at ptrdiff_t minimum and throws
Exceptions::IndexError when out of range. Empty extents reject every index.
Cost is O(1).

```cpp
#include <stratax.h>
#include <array>
#include <cassert>

int main() {
    const stratax::core::Shape shape{2, 3, 4};
    const std::array<std::size_t, 3> indices{1, 0, 2};
    assert(stratax::indexing::offset(shape.strides(), indices) == 14);
    assert(stratax::indexing::normalize_index(-1, 4) == 3);
}
```

Owning Matrix/Tensor operator() uses unchecked offsets. Their at overloads
normalize and validate first. ArrayView operator() validates rank but leaves
component bounds as preconditions. The reserved bounds_checking config flag
does not change these contracts. See @ref arrayview and @ref validation.
