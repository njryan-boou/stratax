@page slice Slice

# Slice {#dev_slice}

Header: `include/stratax/core/Slice.hpp`.

Slice(start, stop, step=1) stores raw signed ptrdiff_t bounds for a half-open
range. The constructor rejects step zero with Exceptions::IndexError. Bounds
are not normalized or clamped until used against an array extent.

start/stop/step return the raw values. size computes the directed range length
with unsigned distance arithmetic, avoiding overflow even at extreme signed
bounds and steps. empty is size == 0. All queries and construction take O(1).
Equality compares all three raw fields; equal selected values do not imply
equal Slice descriptors.

Positive steps select indices below stop, negative steps select indices above
stop. The descriptor supports either sign, but current public slice views reject
negative steps because ArrayView stores unsigned strides.

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    const stratax::core::Slice forward{1, 8, 2};
    const stratax::core::Slice reverse{7, 0, -2};
    assert(forward.size() == 4); // 1, 3, 5, 7
    assert(reverse.size() == 4); // 7, 5, 3, 1; raw descriptor only
    assert(stratax::core::Slice(3, 1).empty());
}
```

Slice owns no elements. See @ref ops_slice for extent normalization, view
ownership, rank preservation, and error contracts.
