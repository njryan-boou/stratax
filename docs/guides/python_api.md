@page python_api "Python API Reference"

# Python API Reference

Concise reference for the public Python API exported from `stratax`.

## Imports

```python
from stratax import (
    Shape,
    Vector,
    Matrix,
    Tensor,
    zeros,
    ones,
    full,
    identity,
    to_vector,
    to_matrix,
    to_tensor,
    sum,
    prod,
    max,
    min,
    argmax,
    argmin,
    mean,
    var,
    std,
)
```

Python containers currently expose `double`-based storage. Slicing, reshaping,
flattening, and conversions return independent containers.

## Shape

| API | Description |
| ----- | ------------- |
| `Shape()` | Create an empty shape. |
| `Shape(other)` | Copy another `Shape`. |
| `Shape([d0, d1, ...])` | Create a shape from dimensions. |
| `shape.rank` | Number of dimensions. |
| `shape.elements` | Product of all dimensions. |
| `shape.empty` | Whether the shape has rank 0. |
| `len(shape)` | Number of dimensions. |
| `shape[i]` | Dimension at index `i`. |
| `list(shape)` | Dimensions as a Python list. |

## Containers

`Vector`, `Matrix`, and `Tensor` share the same basic container operations.

| API | Vector | Matrix | Tensor |
| ----- | -------- | -------- | -------- |
| Empty constructor | `Vector()` | `Matrix()` | `Tensor()` |
| Copy constructor | `Vector(v)` | `Matrix(m)` | `Tensor(t)` |
| Shape constructor | `Vector(Shape([n]))` | `Matrix(Shape([r, c]))` | `Tensor(Shape([...]))` |
| Filled constructor | `Vector(n, value)` | `Matrix(r, c, value)` | `Tensor(shape, value)` |
| Iterable constructor | `Vector([...])` | `Matrix([[...]])` | `Tensor([...])` |
| Size | `.size` | `.size` | `.size` |
| Rank | `.rank` | `.rank` | `.rank` |
| Empty check | `.empty` | `.empty` | `.empty` |
| Shape | `.shape` | `.shape` | `.shape` |
| Strides | `.strides` | `.strides` | `.strides` |
| Fill | `.fill(value)` | `.fill(value)` | `.fill(value)` |
| Convert to lists | `.tolist()` | `.tolist()` | `.tolist()` |
| Reshape | `.reshape(shape)` | `.reshape(shape)` | `.reshape(shape)` |
| Flatten | `.flatten()` | `.flatten()` | `.flatten()` |

`Matrix` also exposes `.rows` and `.cols`.

## Indexing

```python
from stratax import Matrix, Tensor, Vector

v = Vector([1.0, 2.0, 3.0])
v[0] = 10.0
tail = v[1:]

m = Matrix([[1.0, 2.0], [3.0, 4.0]])
value = m[1, 0]
column = m[:, 0]

t = Tensor([2, 2, 2], 1.0)
t[1, 1, 1] = 5.0
last_axis = t[:, :, -1]
```

Negative indexes and slice steps are supported. Slice results are independent
copies.

## Arithmetic

Containers support element-wise arithmetic with matching containers or scalars:

| Operation | Methods |
| ----------- | --------- |
| Addition | `a + b`, `a += b` |
| Subtraction | `a - b`, `a -= b` |
| Multiplication | `a * b`, `a *= b` |
| Division | `a / b`, `a /= b` |
| Unary | `+a`, `-a` |
| Comparison | `a == b`, `a != b` |

Container-to-container operations require compatible shapes. Broadcasting is
planned but not currently implemented.

## Creation Helpers

| Function | Description |
| ---------- | ------------- |
| `zeros(shape)` | Tensor filled with `0.0`. |
| `ones(shape)` | Tensor filled with `1.0`. |
| `full(shape, value)` | Tensor filled with `value`. |
| `identity(size)` | Rank-2 identity tensor with shape `[size, size]`. |

`shape` must be a `Shape`.

## Conversion Helpers

| Function | Description |
| ---------- | ------------- |
| `to_vector(arr)` | Convert a compatible `Vector`, `Matrix`, or `Tensor` to `Vector`. |
| `to_matrix(arr)` | Convert a compatible `Vector`, `Matrix`, or `Tensor` to `Matrix`. |
| `to_tensor(arr)` | Convert a `Vector`, `Matrix`, or `Tensor` to `Tensor`. |

Conversions return new containers.

## Reductions

```python
from stratax import mean, sum

total = sum(arr)
last_axis = sum(arr, -1)
kept = mean(arr, 0, keepdims=True)
```

| Function | Result without `axis` | Result with `axis` |
| ---------- | ----------------------- | -------------------- |
| `sum(arr)` / `sum(arr, axis, keepdims=False)` | `float` | `Tensor` |
| `prod(arr)` / `prod(arr, axis, keepdims=False)` | `float` | `Tensor` |
| `max(arr)` / `max(arr, axis, keepdims=False)` | `float` | `Tensor` |
| `min(arr)` / `min(arr, axis, keepdims=False)` | `float` | `Tensor` |
| `argmax(arr)` / `argmax(arr, axis, keepdims=False)` | `int` | `Tensor` |
| `argmin(arr)` / `argmin(arr, axis, keepdims=False)` | `int` | `Tensor` |
| `mean(arr)` / `mean(arr, axis, keepdims=False)` | `float` | `Tensor` |
| `var(arr)` / `var(arr, axis, keepdims=False)` | `float` | `Tensor` |
| `std(arr)` / `std(arr, axis, keepdims=False)` | `float` | `Tensor` |

Axes may be negative. `keepdims=True` preserves the reduced axis with length 1.

## Exceptions

Stratax exports these exception types:

| Exception | Typical use |
| ----------- | ------------- |
| `StrataxError` | Base exception for Stratax errors. |
| `ShapeError` | Invalid or incompatible shape. |
| `DimensionError` | Invalid dimension count. |
| `IndexError` | Out-of-range index. |
| `TypeError` | Unsupported argument type. |
| `BroadcastError` | Reserved for broadcasting-related shape failures. |
| `ZeroDivisionError` | Division by zero. |
