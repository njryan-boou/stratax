@page vector Vector

# Vector {#dev_vector}

Header: `include/stratax/containers/Vector.hpp`.

`stratax::container::Vector<T>` owns contiguous row-major elements of a
supported DType, including bool. ArrayBase supplies storage, metadata, checked
flat access, and iterators. Default construction produces empty shape `{0}`.

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

Rank is always one, including after a successful move. Shape is `{size()}` and
strides are `{1}`. Size construction value-initializes elements; size/value
fills them; initializer-list construction copies values in list order. The Shape
constructor requires rank one and raises ShapeError otherwise. Vector has no
operator(); use [] or at.

Copy construction deeply copies in O(n). Copy assignment is transactional and
also destroys old storage. Moves transfer the allocation, then leave a distinct
source empty with shape `{0}`. Preparing empty metadata can throw bad_alloc
before transfer; moves are not noexcept. Move construction is O(1), assignment
also destroys old elements. Self-move is unchanged.

```cpp
#include <stratax.h>
#include <cassert>
#include <utility>

int main() {
    stratax::container::Vector<double> values{1.0, 2.0, 3.0};
    values[1] = 5.0;
    assert(values.at(-1) == 3.0);
    auto moved = std::move(values);
    assert(values.rank() == 1 && values.empty());
    assert(moved[1] == 5.0);
}
```

## Construction costs and failures

Allocation failures propagate. Shape/count/stride overflow raises DimensionError;
byte-count overflow may raise std::bad_array_new_length. Construction copies or
initializes n elements plus rank metadata. Matrix nested-list validation also
visits each row, including empty rows. Invalid fixed-rank shapes are rejected
by the container; see the generated header reference for exact constructors.

See @ref arraybase, @ref shape, and @ref arrayview.
