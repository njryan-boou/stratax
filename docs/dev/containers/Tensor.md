@page tensor Tensor

# Tensor {#dev_tensor}

Header: `include/stratax/containers/Tensor.hpp`.

`stratax::container::Tensor<T>` owns contiguous row-major elements of a
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

Shape construction accepts any rank and value-initializes elements; a second
value argument fills them. Rank zero explicitly constructed with Shape{} is
empty, not a scalar. The default Tensor instead has shape `{0}` and rank one.

Copies deeply copy elements and metadata; assignment is transactional. Copy
construction costs O(n + r), and assignment also destroys old state. Noexcept
moves transfer the three buffers, leaving a distinct source empty with rank
zero. Move construction is O(1), assignment also destroys old state.

## Multidimensional indexing

operator() accepts an integral index pack or vector of size_t indices. It checks
neither rank nor bounds. Supply exactly rank nonnegative in-range components;
vector indexing additionally requires positive rank.

Multidimensional at accepts signed indices as a pack or vector. It raises
RankError for the wrong component count and IndexError for an out-of-range
component or empty storage. An empty signed-index vector on a rank-zero Tensor
raises IndexError; a rank mismatch still takes precedence. A single scalar at argument
selects the inherited flat overload. Multidimensional access takes O(r).

```cpp
#include <stratax.h>
#include <cassert>
#include <utility>

int main() {
    stratax::container::Tensor<double> values(stratax::core::Shape{2, 2, 2}, 0.0);
    values(1, 0, 1) = 5.0;
    assert(values.at(-1, 0, -1) == 5.0);
    auto moved = std::move(values);
    assert(values.rank() == 0 && values.empty());
    assert(moved.shape() == stratax::core::Shape({2, 2, 2}));
}
```

## Construction costs and failures

Allocation failures propagate. Shape/count/stride overflow raises DimensionError;
byte-count overflow may raise std::bad_array_new_length. Construction copies or
initializes n elements plus rank metadata. Matrix nested-list validation also
visits each row, including empty rows. Invalid fixed-rank shapes are rejected
by the container; see the generated header reference for exact constructors.

See @ref arraybase, @ref shape, and @ref arrayview.
