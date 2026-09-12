@page dev_stratax stratax.h

# stratax.h

Header: `include/stratax.h`.

The umbrella header includes the public C++ storage, dtype, container,
algorithm, operation, indexing, and printing headers. It does not currently
provide top-level container or algorithm aliases.

| API | Namespace |
| --- | --- |
| Vector, Matrix, Tensor | `stratax::container` |
| Buffer, Shape, Slice, ArrayView, dtype/result traits, math wrappers | `stratax::core` |
| Scalar dtype aliases | `stratax::dtype` |
| zeros, ones, full, identity | `stratax::creation` |
| to_vector, to_matrix, to_tensor, astype | `stratax::conversion` |
| reshape, flatten | `stratax::manipulation` |
| offset, normalize_index, slice | `stratax::indexing` |
| sum, prod, extrema, mean, var, std, axis_reduce | `reduction` |
| Named comparisons, arithmetic/bitwise operators, shape broadcasting helpers | Global namespace |
| Array-array broadcasted_op | `stratax::core` |
| Scalar broadcasted_op overloads | Global namespace |
| StrataxError and derived exceptions | `Exceptions` |

Concepts such as DType and Array are also global. Use qualified names or local
using-declarations. A header-only consumer needs the include directory and
C++20; no Stratax library link step is required.

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    stratax::container::Vector<double> values{1.0, 2.0, 3.0};
    auto tensor = stratax::conversion::to_tensor(values);
    assert(reduction::sum(tensor) == 6.0);
}
```

Component headers can be included individually when a smaller dependency set
is useful. The generated header reference provides exact declarations.
