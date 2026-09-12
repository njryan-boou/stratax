@page print Printing

# Printing {#dev_print}

Header: `include/stratax/io/Print.hpp`; Shape insertion is in `core/Shape.hpp`.

Vector, Matrix, Tensor, and ArrayView stream insertion call the same logical
formatter directly, without converting the input to an owning Tensor. Values
follow row-major logical order, including for noncontiguous views.

## Formatting contract

Empty arrays print `[]` regardless of rank or zero-axis position. Nonempty
arrays use nested brackets. Rank-two siblings are newline-separated without
commas; higher-rank siblings use commas and newlines. Nesting adds four spaces.
Booleans print true/false and int8/uint8 print numerically. Other values use the
stream's scalar formatting. Formatting is for inspection, not a serialization
format that can reconstruct every empty shape.

Shape prints tuple notation, including `(5,)` for rank one and `()` for rank
zero. Array output depends on rank, not whether the type is Matrix or Tensor.
All insertion operators return the stream and do not mutate the array.

```cpp
#include <stratax.h>
#include <sstream>
#include <cassert>

int main() {
    stratax::container::Vector<stratax::dtype::int8> values{1, 2, 3};
    std::ostringstream os;
    os << values;
    assert(os.str() == "[1, 2, 3]");
    std::ostringstream dims;
    dims << stratax::core::Shape{3};
    assert(dims.str() == "(3,)");
}
```

Cost includes scalar output, separator/indentation text, and repeated O(r)
logical-stride construction at recursion nodes. View element lookup adds O(r)
per value. Metadata allocation and configured stream exceptions propagate.
There is no output truncation option. See @ref shape and @ref arrayview.
