@page bitwise Bitwise

# Bitwise {#dev_bitwise}

Header: `include/stratax/ops/Bitwise.hpp`.

## Operations and dtype rules

`~`, `&`, `|`, `^`, `<<`, and `>>` operate on supported Integral dtypes,
excluding bool. Binary operators accept arrays or scalars in either order.
Array-array operations broadcast; matching container families retain their
family and mixed families return Tensor. Allocating helpers require owning
container result traits.

AND, OR, and XOR use the promoted operand dtype. NOT retains the array dtype.
Shifts retain the left value dtype: `array << scalar` keeps the array dtype;
`scalar << array` uses the scalar dtype with the array's container family.
Element operations obey native C++ integer promotion and conversion rules.

## Validation and invariants

Shift counts must be nonnegative and smaller than `sizeof(left_value_type) *
CHAR_BIT`. Invalid counts raise `Exceptions::ValueError`. This limit uses the
stored left dtype, even when C++ would first promote it to int. Scalar counts
are checked even for empty arrays; array counts are checked when used.

Incompatible broadcasting raises `Exceptions::BroadcastError`. An empty operand
cannot supply values to a nonempty result. Compound operators `&=`, `|=`, `^=`,
`<<=`, and `>>=` preserve left shape and dtype and reject shape expansion. They compute from the operands' initial values,
including aliased scalars, overlapping views, and broadcast sources. Repeated
destination offsets receive the last logical row-major result. Compound shifts
validate all used counts before writing; invalid counts leave storage unchanged.
Non-compound results own independent storage.

## Example

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    stratax::container::Vector<stratax::dtype::uint8> flags{3, 5, 6};
    const auto masked = flags & stratax::dtype::uint8{2};
    assert(masked[0] == 2 && masked[1] == 0 && masked[2] == 2);
    flags <<= 1;
    assert(flags[0] == 6);
    try {
        flags <<= 8;
        assert(false);
    } catch (const Exceptions::ValueError&) {}
    assert(flags[0] == 6);
}
```

## Cost and availability

Broadcasted operations take O((n + 1)r), including metadata work. Scalar and
unary owning results take O(n + r). In-place scalar operations on views also
pay O(r) per logical access. Compound operations involving non-owning arrays
stage O(n) results before writing; owning-owning and owning-scalar operations
need no temporary element buffer. Allocating results use O(n + r) storage.
Python does not currently expose these operators.

See @ref broadcasting and @ref concepts.
