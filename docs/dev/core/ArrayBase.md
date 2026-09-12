@page arraybase ArrayBase

# ArrayBase {#dev_arraybase}

Header: `include/stratax/core/ArrayBase.hpp`.

ArrayBase owns a Buffer of elements, a Shape, and a Shape of canonical row-major
strides. Constructors and assignment helpers are protected. Vector, Matrix, and
Tensor add rank constraints and multidimensional interfaces.

## Invariants and access

`size() == shape().elements()` and `rank() == shape().rank() == strides().rank()`.
Nonempty storage is contiguous. Public shape and stride references are read-only.
Metadata queries size/empty/rank/shape/strides take O(1).

Flat `operator[]` is unchecked and requires an index below size. Signed `at(i)`
normalizes negative indices and throws IndexError outside `[-size, size)`.
front/back also throw IndexError on empty arrays. data returns null for empty
owning storage. Forward, const, and reverse iterators traverse the flat sequence;
iterator acquisition is O(1), and full traversal and fill take O(n).

## Construction and assignment

Shape constructors value-initialize elements or copy a fill value, copy the
shape, and compute strides. They take O(n + r) time/storage and propagate
DimensionError for count/stride overflow and allocation/element exceptions.
The buffer-adopting constructor moves the buffer before verifying its count;
on failure the adopted storage is released during unwinding.

Copy construction deeply copies all three buffers. Copy assignment uses
copy-and-swap across the whole state, giving a strong exception guarantee.
Copy assignment costs O(old size + new size + old rank + new rank).
Base moves transfer all buffers and leave a distinct source empty with rank
zero. Base move construction is O(1) and noexcept; move assignment also destroys
old state. Derived Vector/Matrix moves restore fixed-rank empty source metadata
and can allocate, so they are not noexcept.

swap exchanges all buffers in O(1). Pointers/iterators keep referring to the
same allocation under its new owner. Calling base swap must preserve the
receiving derived rank invariants; do not exchange a Vector base with a Matrix
base. Assignment can release destination storage and invalidate its references.

## Checked multidimensional helper

normalized_flat_offset accepts a sized, indexable sequence of signed indices
without caller-supplied messages. It first checks rank with RankError, then
rejects empty storage with IndexError (including rank zero and an empty index
sequence), then checks each component with normalize_index/IndexError. It
combines indices and strides in
O(r). Public multidimensional interfaces belong to the derived containers.

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    stratax::container::Matrix<double> source{{1.0, 2.0}, {3.0, 4.0}};
    auto copy = source;
    copy.fill(9.0);
    assert(source.at(-1) == 4.0);
    assert(copy.size() == copy.shape().elements());
    assert(copy.rank() == copy.strides().rank());
}
```

See @ref buffer, @ref vector, @ref matrix, and @ref tensor. Exact member
signatures are available in the generated ArrayBase reference.
