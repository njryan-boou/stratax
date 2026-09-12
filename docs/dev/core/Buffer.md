@page buffer Buffer

# Buffer {#dev_buffer}

Header: `include/stratax/core/Buffer.hpp`.

Buffer<T, Alignment> owns aligned contiguous storage for a fixed number of
live elements. It is generic over element type, unlike the dtype-constrained
array containers. Alignment defaults to 64 bytes and must be a power of two
at least alignof(T), enforced by static assertions.

## Lifetime and invariants

Empty storage has size zero and a null data pointer. A nonempty buffer owns
exactly size live elements. Allocation uses aligned operator new and matching
operator delete. Constructors clean up partially constructed elements and raw
storage on failure. There is no capacity, resize, insertion, or erasure API;
assignment or swap can replace the element count.

| Operation | Behavior | Cost |
| --- | --- | --- |
| Default constructor | Empty, noexcept | O(1) |
| Size constructor | Value-initializes elements; scalar values become zero | O(n) |
| Size/value constructor | Copy-constructs n copies of the value | O(n) |
| Initializer list | Copies in list order | O(n) |
| Copy constructor | Independent allocation | O(n) |
| Move constructor | Transfers allocation; distinct source becomes empty, noexcept | O(1) |
| Copy assignment | Copy-and-swap; destination unchanged on failure | O(old n + new n) |
| Move assignment | Releases old state and transfers storage; self-move unchanged, noexcept | O(old n) |
| Destructor | Destroys elements and releases storage | O(n) |
| swap | Exchanges allocation and count, noexcept | O(1) |

Byte-count overflow beyond max_size raises std::bad_array_new_length;
allocation raises std::bad_alloc; element constructors may throw.
max_size only bounds byte-count arithmetic, not available memory.

## Access and iterators

size, empty, data, alignment, and max_size are O(1) queries. operator[] is
unchecked and requires index < size. front/back throw Exceptions::IndexError
when empty. fill assigns each element in O(n); a throwing assignment may leave
a modified prefix.

The iterator aliases are T* and const T*, with std::reverse_iterator wrappers.
They provide contiguous random access for live allocations. Empty begin/end
pairs compare equal, and acquiring end avoids arithmetic on a null pointer.
Do not dereference empty iterators or assume an empty data pointer addresses
storage. Iterator arithmetic must stay within the valid allocation range.


Pointers, references, and iterators refer to storage. Move/swap preserve them
under the new owner; replacing or destroying that allocation invalidates them.

```cpp
#include <stratax.h>
#include <cassert>
#include <algorithm>

int main() {
    stratax::core::Buffer<float, 64> samples{1.0F, 2.0F, 3.0F};
    std::fill(samples.begin(), samples.end(), 4.0F);
    auto copy = samples;
    copy.front() = 9.0F;
    assert(samples.front() == 4.0F);
    stratax::core::Buffer<float> empty;
    assert(empty.end() == empty.begin());
}
```

See @ref arraybase and @ref shape. The generated reference lists all const and
reverse iterator accessors and aliases.
