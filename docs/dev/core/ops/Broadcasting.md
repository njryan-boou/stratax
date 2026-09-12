@page broadcasting Broadcasting

# Broadcasting {#dev_broadcasting}

Header: `include/stratax/ops/Broadcasting.hpp`.

## Shape rules

Align dimensions from the right. A pair is compatible if its extents are equal
or either is one; missing leading dimensions behave as ones. The output uses
the non-singleton extent. In particular, zero paired with one produces zero,
not one. Expanded axes always read coordinate zero from the source.

```text
left:    (2, 3, 1)
right:      (1, 4)
result:  (2, 3, 4)
```

`broadcastable(shape1, shape2)` reports compatibility in O(r).
`broadcasted_shape(shape1, shape2)` allocates the common shape or throws
`Exceptions::BroadcastError`. These helpers inspect metadata only: `Shape{}`
has no conflicting axes, but a rank-zero array has no scalar element to read.
Array operations additionally reject an empty operand supplying a nonempty
result. They can produce an empty result when the shapes are compatible.

## Traversal and result selection

`broadcasted_op(lhs, rhs, op)` calls op in logical result order without changing
its operands. Two numeric arrays infer a promoted result dtype. The explicit
`stratax::core::broadcasted_op<Result>(lhs, rhs, op)` overload selects a dtype,
also allowing Boolean results. Same-family arrays preserve their family;
mixed families produce Tensor through PromoteArray.

Array-scalar and scalar-array forms preserve array shape and family and infer
the dtype from both operands. Allocating forms require the owning result
traits. The callable receives original element types, not values converted to
the result dtype. Callable and allocation exceptions propagate.

## Example

```cpp
#include <stratax.h>
#include <cassert>
#include <functional>

int main() {
    stratax::container::Matrix<int> column{{1}, {2}};
    stratax::container::Vector<int> row{10, 20, 30};
    auto result = stratax::core::broadcasted_op(column, row, std::plus<>{});
    assert(result.shape() == stratax::core::Shape({2, 3}));
    assert(result(1, 2) == 32);
    auto mask = stratax::core::broadcasted_op<bool>(column, row, std::less<>{});
    assert(mask(0, 0));
    assert(broadcasted_shape(stratax::core::Shape{0, 3}, stratax::core::Shape{1, 3})
           == stratax::core::Shape({0, 3}));
}
```

## Cost

With n output elements and maximum rank r, array-array traversal takes
O((n + 1)r); scalar forms take O(n + r). Results allocate O(n + r) storage.
Broadcasting is shared by arithmetic, comparison, bitwise, and binary math
operations. Compound operators separately require the output shape to match
the left operand exactly.

See @ref arithmetic, @ref comparison, @ref bitwise, and @ref math.
