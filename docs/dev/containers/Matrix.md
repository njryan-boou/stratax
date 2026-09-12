@page matrix Matrix

# Matrix {#dev_matrix}

Header: `include/stratax/containers/Matrix.hpp`.

`stratax::container::Matrix<T>` owns contiguous row-major elements of a
supported DType, including bool. ArrayBase supplies storage, metadata, checked
flat access, and iterators. Default construction produces empty shape `{0, 0}`.

## Common interface and invariants

`size() == shape().elements()` and `rank() == shape().rank() == strides().rank()`.
Metadata size/empty/rank/shape/strides queries take O(1). Shape and strides are
read-only. `operator[]` is unchecked flat indexing; `at(i)` checks a signed flat
index, including negative indices. Invalid at, front, or back access raises
Exceptions::IndexError. data returns null for empty owning storage.

begin/end, const variants, and reverse iterators traverse flat row-major values;
acquisition is O(1), traversal and fill are O(n). Owning iterators are contiguous.
Member and ADL swap exchange storage and metadata in O(1). Pointers and iterators
follow the allocation under its new owner; replacing or destroying storage
invalidates its references. Slicing is a free function returning ArrayView.


## Rank, construction, and moves

Rank stays two, including after a successful move. rows and cols query the two
dimensions in O(1); strides are `{cols(), 1}`, including zero columns. Empty
shapes `{0, n}` and `{n, 0}` are valid. Dimensions constructors value-initialize
or fill elements. Shape construction requires rank two (ShapeError otherwise).
Nested initializer lists must be rectangular (ShapeError otherwise); empty rows
retain row count and zero columns.

operator(row, col) checks neither component. Both must be in range. at(row, col)
checks signed components independently, including negatives, in O(1). Inherited
one-argument at remains checked flat access.

Copies own independent storage and assignment has a strong exception guarantee.
Moves transfer storage and leave a distinct source at `{0, 0}`. Empty metadata
allocation may fail before transfer, so moves are not noexcept. Move construction
is O(1); assignment also destroys old storage. Self-move is unchanged.

```cpp
#include <stratax.h>
#include <cassert>
#include <utility>

int main() {
    stratax::container::Matrix<double> values{{1.0, 2.0}, {3.0, 4.0}};
    values(1, 0) = 8.0;
    assert(values.at(-1, 0) == 8.0);
    auto moved = std::move(values);
    assert(values.rows() == 0 && values.cols() == 0 && values.rank() == 2);
    assert(moved.at(-1) == 4.0);
}
```

## Construction costs and failures

Allocation failures propagate. Shape/count/stride overflow raises DimensionError;
byte-count overflow may raise std::bad_array_new_length. Construction copies or
initializes n elements plus rank metadata. Matrix nested-list validation also
visits each row, including empty rows. Invalid fixed-rank shapes are rejected
by the container; see the generated header reference for exact constructors.

See @ref arraybase, @ref shape, and @ref arrayview.
