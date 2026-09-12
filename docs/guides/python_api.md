@page python_api Python API Reference

# Python API Reference

The public package re-exports the compiled `_core` API. Numeric Vector, Matrix,
and Tensor store double values. Boolean comparison results and int64 index
results have separate classes with narrower interfaces.

## Construction and metadata

| Type | Constructors |
| --- | --- |
| Shape | `Shape()`, `Shape(other)`, `Shape(size)`, `Shape([d0, d1, ...])` |
| Vector | `Vector()`, `Vector(other)`, `Vector(size)`, `Vector(size, value)`, `Vector(shape)`, `Vector(values)` |
| Matrix | `Matrix()`, `Matrix(other)`, `Matrix(rows, cols)`, `Matrix(rows, cols, value)`, `Matrix(shape)`, `Matrix(nested_rows)` |
| Tensor | `Tensor()`, `Tensor(other)`, `Tensor(shape)`, `Tensor(shape, value)` |

Tensor accepts a Shape or an iterable of **dimensions**, not nested element
data. `Tensor([2, 3])` creates six zeros. Matrix rows must be rectangular.
Dimensions must be nonnegative integers. Checked scalar construction and element
assignment reject bool and nonfinite values; arithmetic results follow core
floating-point rules and can be nonfinite.

Shape exposes rank, elements, empty, strides, dimension indexing, iteration, and
equality. Shape(size) creates a rank-one shape. Shape.strides computes row-major
element strides as another Shape.
Shape.empty means rank zero. Shape.elements is zero for rank zero or any zero
extent. Owning containers expose size, rank, empty, shape, strides, fill(value),
tolist(), reshape(shape), and flatten(); Matrix adds rows and cols. Strides are
in elements. Metadata are properties, not method calls.

Owning numeric `len(arr)` is the first extent, or zero for rank zero; iteration
visits all elements in flat logical order. tolist returns nested lists matching
the shape as far as a list can express it. A zero outer dimension becomes `[]`
and cannot preserve unseen trailing dimensions in the list representation.
Default Vector/Tensor have shape `[0]`; default Matrix has shape `[0, 0]`.
Explicit `Tensor(Shape())` is empty rank zero, not a scalar.

```python
from stratax import Tensor, Shape

arr = Tensor([2, 3], 4.0)
assert arr.shape == Shape([2, 3])
assert arr.shape.strides == Shape([3, 1])
assert Shape(3) == Shape([3])
assert len(arr) == 2 and arr.size == 6
assert list(arr) == [4.0] * 6
assert arr.tolist() == [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]
```

## Indexing and slice views

Vector integer access is flat. Matrix accepts `(row, col)` for read/write and
an integer for flat reads; Matrix assignment requires a row/column tuple.
Tensor supports flat integers and full-rank integer tuples. Negative indices
count from the end. Rank-zero arrays have no addressable elements: reading or
writing with an empty tuple raises `stratax.IndexError`. A tuple whose length
differs from the array rank raises `stratax.RankError`.

Positive-step slices return shared ArrayView. Vector takes one slice, Matrix
supports row slicing or a pair of indices/slices, and Tensor takes one entry
per axis (a bare slice works only at rank one). Integer entries mixed with
slices retain length-one axes. Slice bounds clamp to the extents. Negative
steps raise stratax.IndexError; Python rejects a zero slice step with built-in
ValueError. Bulk slice assignment and advanced indexing are not bound.

```python
from stratax import Matrix, Shape, Tensor

owner = Matrix([[1.0, 2.0], [3.0, 4.0]])
view = owner[:, 0]
assert view.shape == Shape([2, 1])
view[1, 0] = 9.0
assert owner[1, 0] == 9.0
copy = Tensor(view.shape)
for i in range(view.size):
    copy[i] = view[i]
assert copy.tolist() == [[1.0], [9.0]]
```

Views keep their owner alive and expose size, rank/ndim, empty, shape, strides,
tolist(), and checked integer/tuple reads and writes. len(view) counts all
selected elements. Views have no public constructor, further slicing, arithmetic,
reshape, conversion, or reduction overloads. Use an explicit owning copy when
an operation needs a numeric container.

## Arithmetic and comparison

`+`, `-`, `*`, `/`, unary +/- and compound arithmetic accept owning numeric
containers; binary forms also accept scalars in either order and mixed container
families. Array-array operations broadcast trailing dimensions. Matching
families retain their family; mixed families return Tensor. Compound operations
preserve the left object's shape and dtype and reject expansion with
BroadcastError. Compound division validates all used divisors before writing.
Scalar division by zero raises ZeroDivisionError even for an empty input.
An empty rank-zero operand cannot supply values to a nonempty result.

`==`, `!=`, `<`, `<=`, `>`, and `>=` return BoolVector, BoolMatrix, or BoolTensor
masks with the broadcasted shape. Named functions equal, not_equal, less,
less_equal, greater, and greater_equal expose the same operations. At least one
operand must be a supported numeric array. These are element-wise comparisons,
not whole-array equality or approximate equality.

```python
from stratax import Vector

values = Vector([1.0, 2.0, 3.0])
mask = values >= 2.0
assert mask.tolist() == [False, True, True]
assert not all(mask)
```

Boolean result classes expose metadata, fill, tolist, repr, flat iteration, and
len (total element count). They do not expose public constructors or indexed
access. Their truth value is not a reduction of the Boolean elements; use
Python all/any over iteration for a whole-mask check.

## Creation and copies

| Function | Output selection |
| --- | --- |
| `zeros(shape)`, `ones(shape)`, `full(shape, value)` | Tensor; shape must be a Shape |
| `zeros(size)`, `ones(size)`, `full(size, value)` | Vector |
| `zeros(rows, cols)`, `ones(rows, cols)`, `full(rows, cols, value)` | Matrix |
| `identity(size)` | Square Matrix |
| `to_vector(arr)` | Vector-compatible shape: rank one or exactly one extent different from one |
| `to_matrix(arr)` | Matrix-compatible shape: rank two or exactly two extents different from one |
| `to_tensor(arr)` | Tensor with unchanged shape |
| `arr.reshape(shape)` | Tensor with equal element count |
| `arr.flatten()` | Vector in flat logical order |

Creation, conversion, reshape, and flatten allocate independent storage.
Conversion accepts owning numeric arrays. Only extent-one axes are removed;
zero extents survive. Reshape does not infer -1 dimensions.

## Reductions

All nine reductions accept owning numeric arrays. Use `(arr)` globally or
`(arr, axis, keepdims=False)` for an axis; axis=None and multi-axis sequences are
not bound. Axes may be negative. keepdims retains extent one; reducing rank one
without keepdims still returns shape `[1]`.

| Functions | Global result | Axis result | Empty reduction slice |
| --- | --- | --- | --- |
| sum | float | Tensor | 0 |
| prod | float | Tensor | 1 |
| min, max | float | Tensor | stratax.IndexError |
| argmin, argmax | int | IndexTensor | stratax.IndexError |
| mean, var, std | float | Tensor | stratax.ZeroDivisionError |

An empty output domain produces an empty result without reducing any slices.
Extrema use the first tied position. Global indices are flat; axis indices are
positions along that axis. IndexTensor stores int64 and exposes metadata, fill,
tolist, iteration, membership, len (first extent), and checked flat/tuple reads.
It has no public constructor or indexed assignment. Variance/std are population
statistics dividing by n, with no ddof or NaN-skipping option.

```python
from stratax import Matrix, Shape, argmax, mean, sum

arr = Matrix([[1.0, 2.0], [3.0, 4.0]])
assert sum(arr) == 10.0
assert sum(arr, -1).tolist() == [3.0, 7.0]
assert mean(arr, 0, keepdims=True).shape == Shape([1, 2])
assert argmax(arr, 1).tolist() == [1, 1]
```

## Exceptions and unbound APIs

StrataxError derives from built-in RuntimeError. Exported subclasses are
ShapeError, DimensionError, RankError, IndexError, TypeError, BroadcastError,
ZeroDivisionError, AxisError, OverflowError, and ValueError. RankError handles
wrong index/slice counts; AxisError handles invalid reduction axes;
DimensionError includes shape arithmetic overflow.

These are distinct from Python's built-in exceptions of the same names.
Binding conversion overflow and Shape.elements overflow explicitly raise
built-in OverflowError. Shape.strides overflow raises stratax.DimensionError. Python
slice parsing and pybind11 argument matching may raise built-in ValueError or
TypeError. C++ allocation failures may become MemoryError. Catch the documented
category at the relevant call, rather than assuming every error is StrataxError.

C++ bitwise operators, standard math wrappers, astype, and most additional
dtypes are not exported by the Python package. See @ref user_guide and
@ref dev_bindings for implementation boundaries.

## Empty-input example

```python
import stratax as sx

empty = sx.Vector()
assert sx.sum(empty) == 0.0
assert sx.prod(empty) == 1.0
for operation, error in [(sx.max, sx.IndexError), (sx.mean, sx.ZeroDivisionError)]:
    try:
        operation(empty)
    except error:
        pass
    else:
        raise AssertionError("An undefined empty reduction must raise")
assert sx.sum(sx.Tensor([0, 3]), 0).tolist() == [0.0, 0.0, 0.0]
assert sx.max(sx.Tensor([0, 3]), 1).empty
```
