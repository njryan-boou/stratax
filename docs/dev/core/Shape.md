@page shape Shape

# Shape {#dev_shape}

Header: `include/stratax/core/Shape.hpp`.

Shape owns unsigned dimensions in outermost-to-innermost order. It stores
metadata, not array values, and computes element counts and canonical row-major
strides. Public dimension access and iterators are read-only.

## Construction and ownership

Default construction produces rank zero. Initializer-list and vector
constructors copy dimensions, including zero extents. They do not compute every
possible shape product during construction. Copying duplicates dimension
storage; copy assignment has Buffer's strong exception guarantee. Moving leaves
a distinct source rank zero; self-move leaves it unchanged. swap exchanges
storage without copying and preserves references under the new owner.

## Queries and errors

| API | Behavior | Cost |
| --- | --- | --- |
| rank | Number of stored dimensions | O(1) |
| empty | Whether rank is zero, not whether elements is zero | O(1) |
| elements | Product, or zero for rank zero/any zero extent | O(r) |
| strides | New Shape; last stride one, earlier strides are right-side products | O(r) |
| operator[] | Unchecked dimension access | O(1) |
| at | Checked signed access with negative indices | O(1) |
| begin/end and const/reverse variants | Read-only contiguous traversal | O(1) acquisition |
| ==, != | Same rank and dimensions | O(r) worst case |

An invalid at index raises Exceptions::IndexError. elements detects nonzero
product overflow with DimensionError and returns zero if any extent is zero,
even when an earlier prefix would overflow. strides checks each right-side
product separately and can throw even for a shape with zero total elements.
Both zero rank and zero extents are valid. No inferred negative dimensions exist.

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    const stratax::core::Shape shape{2, 3, 4};
    assert(shape.elements() == 24);
    assert(shape.strides() == stratax::core::Shape({12, 4, 1}));
    assert(shape.at(-1) == 4);
    const stratax::core::Shape zero{2, 0, 4};
    assert(!zero.empty() && zero.elements() == 0);
    assert(stratax::core::Shape{}.empty());
}
```

Shape prints tuple notation: `(2, 3, 4)`, `(5,)`, or `()`. Construction/copying
and allocation cost O(r); copy assignment also destroys old metadata. Allocation
failures propagate. Move construction and swap are O(1).
