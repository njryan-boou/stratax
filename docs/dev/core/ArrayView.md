@page arrayview ArrayView

# ArrayView {#dev_arrayview}

Header: `include/stratax/core/ArrayView.hpp`.

`stratax::core::ArrayView<T>` holds a pointer and copies of shape and unsigned
element-stride metadata. It never owns the allocation or extends its lifetime.
Copies alias the same elements. Strides can describe gaps, transposes, and
repeated elements through zero strides; negative strides are not representable.

## Construction invariants

The constructor takes `(data, shape, strides)` and validates equal metadata
ranks, representable element count and maximum reachable offset, and a non-null
pointer for nonempty storage. It cannot verify the external allocation size or
lifetime: the caller must provide live storage for every reachable element.
Rank mismatch raises RankError, a nonempty null pointer raises ValueError,
and arithmetic overflow raises DimensionError. Metadata allocation can fail.
An empty view may have a non-null pointer; rank zero is empty.

## Access and constness

| API | Contract | Cost |
| --- | --- | --- |
| `shape()`, `strides()`, `rank()`, `ndim()`, `data()` | Metadata or base pointer | O(1) |
| `size()`, `empty()` | Compute element count from shape | O(r) |
| `operator[](i)` | Unchecked logical flat access; i must be below size | O(r) |
| `at(i)` | Checked flat signed index, including negatives | O(r) |
| `operator()(indices...)` | Checks rank; caller ensures components are in bounds | O(r) |
| `at(vector_of_signed_indices)` | Checks rank and components; empty storage raises IndexError | O(r) |
| `as_const()` | Copies metadata and exposes const elements | O(r) |
| `begin()`, `end()` | Logical random-access iterators | O(1), O(r) respectively |

Flat order is row-major logical order, not physical contiguous order. T may be
const. Const access to a view also returns const element references, but another
mutable alias may still modify the allocation. Checked invalid indices raise
IndexError, including an empty index vector on a rank-zero view; mismatched
multidimensional index counts raise RankError before empty-storage checks.

Iterators retain a pointer to the view object, so both that object and its
storage must outlive use. Dereference takes O(r); arithmetic is O(1) and must
stay within the valid range. Ordering and subtraction require the same view.
Iterators are not contiguous.

## Compound mutation

C++ arithmetic and bitwise compound operators preserve the view layout and
compute from the operands' values before mutation. This includes overlapping
views and scalar references into the destination. Results are staged in O(n)
temporary storage. Repeated physical offsets receive the last logical row-major
write, including repetition caused by nonzero strides. See @ref arithmetic and
@ref bitwise for validation and callable exception guarantees.

## Example

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    int storage[]{0, 1, 2, 3, 4, 5};
    stratax::core::ArrayView<int> transposed(storage,
        stratax::core::Shape{3, 2}, stratax::core::Shape{1, 3});
    assert(transposed[1] == 3);
    assert(transposed[2] == 1);
    transposed.at(-1) = 9;
    assert(storage[5] == 9);
    auto owned = stratax::conversion::to_tensor(transposed);
    assert(owned(2, 1) == 9);
}
```

The Array concept accepts views, but allocating arithmetic/math/comparison
helpers also need owning result traits. C++ conversions, reshape, flatten, and
reductions accept views. Python exposes a narrower owner-retaining wrapper;
see @ref python_api.
