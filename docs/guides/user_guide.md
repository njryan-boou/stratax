@page user_guide User Guide

# User Guide

Stratax provides owning arrays and shared slice views. The examples below use
`stratax.h` in C++ and the public `stratax` package in Python.

## Containers and access

`Vector` has rank one, `Matrix` has rank two, and `Tensor` accepts arbitrary
rank. Owning arrays store values contiguously in row-major order. Their element
count equals the product of their dimensions, except that rank zero represents
an empty array with zero elements.

C++ `operator[]` is unchecked flat access. Matrix and Tensor `operator()` are
unchecked multidimensional access; Vector has no `operator()`. Use `at()` for
checked signed indices, including negative indices counted from the end.
Python scalar and tuple indexing checks bounds.

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    stratax::container::Vector<double> values{1.0, 2.0, 3.0};
    values[1] = 5.0;
    assert(values.at(-1) == 3.0);
    const auto doubled = values * 2.0;
    assert(doubled[1] == 10.0);
}
```

## Slices share storage

Positive-step slices return `ArrayView`, preserving every selected axis.
Writing an element of a mutable view changes the source. C++ callers must keep
the underlying allocation alive; Python views retain their owner. Negative-step
views are currently rejected, even though a C++ `Slice` can describe a reverse
range. Zero steps are invalid.

```python
from stratax import Matrix, Shape

matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])
column = matrix[:, 0]
assert column.shape == Shape([2, 1])
column[1, 0] = 9.0
assert matrix[1, 0] == 9.0
```

C++ slicing takes explicit signed bounds and clamps them to each extent:

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    stratax::container::Vector<double> values{1.0, 2.0, 3.0, 4.0};
    auto view = stratax::indexing::slice(values, stratax::core::Slice{0, 4, 2});
    view[1] = 8.0;
    assert(values[2] == 8.0);
    auto copy = stratax::conversion::to_vector(view);
    copy[0] = 7.0;
    assert(values[0] == 1.0);
}
```

Python views currently support metadata, `tolist()`, and scalar or integer-tuple
access. View slicing, arithmetic, conversions, and reductions are not bound.
To materialize a Python view, construct `Tensor(view.shape)` and assign its flat
elements from `view[i]`, as shown in @ref python_api.

## Broadcasting and result types

Arithmetic and comparisons align dimensions from the right. Each pair must be
equal or contain a dimension of length one. Mixed C++ owning container families
produce a Tensor; same-family operations retain that family. C++ result dtypes
follow the promotion traits. Element operators run on the original operand
types before result conversion, so result promotion does not guarantee safe
intermediate integer arithmetic.

```python
from stratax import Matrix, Shape, BroadcastError

column = Matrix([[1.0], [2.0]])
row = Matrix([[10.0, 20.0, 30.0]])
assert (column + row).tolist() == [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]
assert (column < row).shape == Shape([2, 3])
try:
    column += row
except BroadcastError:
    pass
else:
    raise AssertionError("Compound assignment must preserve the left shape")
assert column.shape == Shape([2, 1])
```

Compound assignment preserves the left shape and dtype; it rejects broadcasting
that would expand that shape. Division checks used divisors for zero. Dividing
by a scalar zero raises even for an empty array. An empty rank-zero operand
cannot supply values to a nonempty broadcast result.

## Reshape, flatten, and conversion

Reshape copies into a Tensor with the requested element count. Flatten copies
into a Vector in logical row-major order. Neither operation shares storage or
accepts inferred `-1` dimensions.

```python
from stratax import Shape, Tensor

tensor = Tensor(Shape([2, 2, 2]), 1.0)
reshaped = tensor.reshape(Shape([4, 2]))
flat = tensor.flatten()
assert reshaped.shape == Shape([4, 2])
assert flat.tolist() == [1.0] * 8
```

`to_tensor` preserves shape. `to_vector` accepts rank one or exactly one extent
different from one; `to_matrix` accepts rank two or exactly two extents different
from one. Removed axes have extent one, so zero extents survive conversion.
All these conversions copy. C++ accepts views as conversion inputs; Python
conversion helpers accept the owning double containers.

## Creation

For `zeros`, `ones`, and `full`, a Shape selects Tensor, one integer size selects
Vector, and two dimensions select Matrix. `identity` returns Matrix. C++ callers
specify a supported dtype, including bool; Python helpers create double arrays.

```python
from stratax import Shape, full, identity, ones, zeros

assert zeros(Shape([2, 3])).shape == Shape([2, 3])
assert ones(3).tolist() == [1.0, 1.0, 1.0]
assert full(2, 2, 7.0).tolist() == [[7.0, 7.0], [7.0, 7.0]]
assert identity(2).tolist() == [[1.0, 0.0], [0.0, 1.0]]
```

## Reductions and empty arrays

Global reductions return scalars. Axis reductions return Tensor, or IndexTensor
for Python argmin/argmax. Axes may be negative. `keepdims=True` retains the axis
with extent one; without it, reducing rank one still returns shape `[1]`
because rank-zero arrays have no scalar storage. C++ exposes the same convention
through the `reduction` namespace.

```python
from stratax import Matrix, Shape, argmax, mean, sum

arr = Matrix([[1.0, 2.0], [3.0, 4.0]])
assert sum(arr) == 10.0
assert sum(arr, -1).tolist() == [3.0, 7.0]
assert mean(arr, 0, keepdims=True).shape == Shape([1, 2])
assert argmax(arr, 1).tolist() == [1, 1]
```

Empty sums return zero and empty products return one. Empty min/max/argmin/argmax
raise `IndexError`; empty mean/var/std raise `ZeroDivisionError`. For axis
reductions these rules apply to each produced output slice. If there are no
output positions, the result is empty and no slice reduction runs.
Variance and standard deviation are population statistics, with no `ddof` option.
C++ integer sums/products do not automatically widen accumulation.

## Current scope

C++ includes bitwise and standard-library math wrappers. These are not exported
by the Python package. Python numeric containers use double; comparison results
use Boolean containers and axis extrema indices use int64 storage. Linear
algebra, random sampling, broader statistics, and file I/O remain planned.

See @ref python_api for binding details and @ref roadmap for planned work.
