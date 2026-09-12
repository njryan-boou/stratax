@page conversions Conversions

# Conversions {#dev_conversions}

Header: `include/stratax/algorithms/Conversion.hpp`.
Functions live under `stratax::conversion`.

## Shape contracts

| Function | Accepted shape | Result |
| --- | --- | --- |
| `to_vector(arr)` | Rank one, or exactly one extent different from one | Vector with source element count |
| `to_matrix(arr)` | Rank two, or exactly two extents different from one | Matrix with original rank-two shape or the two retained extents |
| `to_tensor(arr)` | Any valid source shape | Tensor with identical shape |

Each conversion copies values in logical row-major order into independent
storage, including for strided C++ views. Only singleton extents are removed:
`{1, 0, 3}` converts to Matrix `{0, 3}`. A higher-rank shape containing only ones
is not vector- or matrix-compatible under these rules. Rank-zero arrays cannot
convert to Vector/Matrix through these helpers; flatten provides a Vector copy.
Invalid conversion shapes raise `Exceptions::ShapeError`. Shape arithmetic and
allocation errors propagate.

## Dtype casts

`astype<To>(arr)` preserves the owning container family and shape and converts
each element with `static_cast<To>`. Both dtypes must be supported and the cast
must be valid. Narrowing and precision loss are not checked. The supplied
RebindArray specializations cover owning containers, not ArrayView.
Python does not currently bind astype.

## Example

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    stratax::container::Tensor<double> source(stratax::core::Shape{1, 3, 1}, 2.5);
    auto vector = stratax::conversion::to_vector(source);
    auto ints = stratax::conversion::astype<int>(vector);
    assert(vector.size() == 3 && ints[0] == 2);
    vector[0] = 8.0;
    assert(source[0] == 2.5);
    stratax::container::Tensor<int> empty(stratax::core::Shape{1, 0, 3});
    assert(stratax::conversion::to_matrix(empty).shape() == stratax::core::Shape({0, 3}));
}
```

## Cost

Owning inputs take O(n + r) time; view inputs take O((n + 1)r), because logical
access computes strided offsets. Output storage is O(n + r). Shape compatibility
helpers in `conversion::detail` are implementation details.
Python conversion functions accept the owning double Vector, Matrix, and Tensor
classes; see @ref python_api.
