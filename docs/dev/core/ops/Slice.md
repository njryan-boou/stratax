@page ops_slice Slice Operations

# Slice Operations {#dev_ops_slice}

Header: `include/stratax/indexing/Slicing.hpp`.

## Shared storage and lifetime

`stratax::indexing::slice` returns ArrayView with copied metadata and shared
elements. Mutable sources yield mutable views; const owning sources yield
read-only elements. The caller must keep the source allocation alive. Destroying
or assigning the owner can invalidate a view; moving or swapping storage changes
which object must keep that allocation alive.

Vector takes one Slice, Matrix takes row and column Slices, and Tensor takes
one Slice per axis either as a parameter pack or a vector of Slices. C++ also
supports slicing an ArrayView with a vector of Slices. All axes are retained,
including length-one and zero axes. No selected elements are copied.

## Normalization and errors

Bounds are signed and half-open. Negative bounds count from the end and
positive-step bounds clamp to `[0, extent]`. Directionally empty selections
produce empty views with the fully selected shape. Slice construction rejects
step zero. Public view-producing overloads reject negative steps with
`Exceptions::IndexError`, because their strides are unsigned.

The internal normalization helper can describe reverse ranges, including its
special stop=-1 sentinel. That does not enable reverse views in the public API.
Extents that do not fit ptrdiff_t raise IndexError. Tensor/view slice-count
mismatches raise RankError; checked offset/stride overflow raises DimensionError.
Metadata allocation can fail. For empty results the base pointer is retained
without shifting it beyond the source allocation; it need not be null.

## Example

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    stratax::container::Matrix<int> source{{0, 1, 2}, {3, 4, 5}};
    auto view = stratax::indexing::slice(source,
        stratax::core::Slice{0, 2}, stratax::core::Slice{0, 3, 2});
    assert(view.shape() == stratax::core::Shape({2, 2}));
    assert(view.strides() == stratax::core::Shape({3, 2}));
    view.at(1) = 20;
    assert(source(0, 2) == 20);
    auto copy = stratax::conversion::to_matrix(view);
    copy(0, 0) = 99;
    assert(source(0, 0) == 0);
}
```

Construction takes O(r) time and metadata storage for rank r, independent of
selected element count. Logical flat access takes O(r). Vector and Matrix
slice construction have fixed rank and O(1) metadata cost.
Python slice views keep their owner alive but do not yet support further
slicing; see @ref python_api and @ref arrayview.
