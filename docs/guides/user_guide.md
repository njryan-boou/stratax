@page user_guide "User Guide"

# User Guide

Practical notes for using the current Stratax API from C++ and Python.

## Containers

Stratax centers on `Vector`, `Matrix`, and `Tensor`.

```cpp
#include <stratax.h>

stratax::Vector<double> values{1.0, 2.0, 3.0};
auto doubled = values * 2.0;
```

```python
from stratax import Vector

values = Vector([1.0, 2.0, 3.0])
doubled = values * 2.0
```

Use `Vector` for one-dimensional data, `Matrix` for two-dimensional data, and
`Tensor` for N-dimensional data. `Shape` stores dimension metadata for reshape
and tensor construction.

## Indexing and Slicing

C++ uses `operator()` for multidimensional access and `slice(...)` helpers for
slicing.

Core container types use `operator[]` for flat indexing making container types share algorithm logic.

```cpp
stratax::Vector<double> values{1.0, 2.0, 3.0, 4.0};
values(1) = 5.0;
auto every_other = stratax::slice(values, stratax::Slice{0, values.size(), 2});
```

Python uses familiar item access and slice syntax.

```python
from stratax import Matrix

matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])
matrix[1, 0] = 9.0
first_column = matrix[:, 0]
reversed_rows = matrix[::-1, :]
```

Negative indexing is supported in both APIs. Slicing currently returns
copy-based results.

## Shape, Reshape, and Flatten

Use `Shape` when constructing tensors or changing logical dimensions.

```cpp
stratax::Tensor<double> tensor(stratax::Shape{2, 2, 2}, 1.0);
auto matrix_like = stratax::reshape(tensor, stratax::Shape{4, 2});
auto flat = stratax::flatten(tensor);
```

```python
from stratax import Shape, Tensor

tensor = Tensor(Shape([2, 2, 2]), 1.0)
matrix_like = tensor.reshape(Shape([4, 2]))
flat = tensor.flatten()
```

Reshape requires the total element count to stay compatible with the source
container.

## Creation and Conversion

Creation helpers provide common initialized containers.

```cpp
auto zeros_tensor = stratax::zeros<double>(stratax::Shape{2, 3});
auto eye = stratax::identity<double>(3);
```

```python
from stratax import Shape, full, identity, ones, zeros

zeros_tensor = zeros(Shape([2, 3]))
ones_vector = ones(4)
full_matrix = full(2, 3, 7.0)
eye = identity(3)
```

Conversions move between compatible container shapes.

```cpp
auto tensor = stratax::to_tensor(values);
auto roundtrip = stratax::to_vector(tensor);
```

```python
from stratax import to_tensor, to_vector

tensor = to_tensor(values)
roundtrip = to_vector(tensor)
```

## Reductions

Reductions are available globally and by axis.

```cpp
auto total = stratax::sum(tensor);
auto last_axis = stratax::reductions::sum(tensor, -1);
auto mean_axis0 = stratax::reductions::mean(tensor, 0, true);
```

```python
from stratax import mean, sum

total = sum(tensor)
last_axis = sum(tensor, -1)
mean_axis0 = mean(tensor, 0, keepdims=True)
```

Python reductions support `keepdims` for preserving rank.

## Python API Notes

Python bindings currently expose `double` container specializations. Use
`tolist()` for quick inspection or for handing values to ordinary Python code.

```python
from stratax import Vector

values = Vector([1.0, 2.0, 3.0])
print(values.tolist())
```

For a compact list of Python constructors, helpers, reductions, and exceptions,
see @ref python_api "Python API Reference".

## Current Limits

Non-owning views, linear algebra, random sampling, statistics, and file I/O are
planned areas. Arithmetic broadcasting is available; comparisons remain
shape-sensitive. Check @ref roadmap "Roadmap" for the current direction.
