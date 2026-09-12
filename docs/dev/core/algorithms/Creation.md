@page creation Creation

# Creation {#dev_creation}

Header: `include/stratax/algorithms/Creation.hpp`.

## Output selection

Functions are available under `stratax::creation`. T must satisfy DType, which includes bool.

| Call | Owning result | Initialization |
| --- | --- | --- |
| `zeros<T>(shape)` / `ones<T>(shape)` | Tensor | `T{}` / `T{1}` |
| `zeros<T>(size)` / `ones<T>(size)` | Vector | `T{}` / `T{1}` |
| `zeros<T>(rows, cols)` / `ones<T>(rows, cols)` | Matrix | `T{}` / `T{1}` |
| `full<T>(shape, value)` | Tensor | Copies value |
| `full<T>(size, value)` | Vector | Copies value |
| `full<T>(rows, cols, value)` | Matrix | Copies value |
| `identity<T>(size)` | Matrix | Ones on diagonal, zeros elsewhere |

## Invariants and failures

Shape overloads preserve all dimensions, including zero extents. Rank-zero
Shape creates an empty Tensor. Identity has shape `{size, size}`; size zero
produces an empty rank-two Matrix. All results own their storage.

Shape element/stride overflow raises `Exceptions::DimensionError`. Allocation
can raise `std::bad_alloc` or `std::bad_array_new_length` for an unrepresentable
storage byte count. Python dimensions are checked before unsigned conversion.

## Example

```cpp
#include <stratax.h>
#include <cassert>
#include <type_traits>

int main() {
    auto z = stratax::creation::zeros<double>(stratax::core::Shape{2, 3});
    auto o = stratax::creation::ones<float>(4);
    auto f = stratax::creation::full<int>(2, 2, 7);
    auto eye = stratax::creation::identity<bool>(3);
    static_assert(std::is_same_v<decltype(eye), stratax::container::Matrix<bool>>);
    assert(z.size() == 6 && o[0] == 1.0F && f(1, 1) == 7);
    assert(eye(1, 1) && !eye(1, 2));
}
```

## Cost

Initialization takes O(n + r) time and storage for n elements and rank r.
Identity takes O(size squared) element work. Python offers the same overload
selection for double containers; see @ref python_api.
