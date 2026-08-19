@page reductions Reductions

# Reductions {#dev_reductions}

Version: v0.2.0

Status: Complete

Header: `include/stratax/algorithms/Reductions.hpp`

---

## Overview

`Reductions.hpp` implements scalar and axis-based reduction algorithms for vectors, matrices, and tensors.

All reduction functions operate on containers and return either scalar values (for global reductions) or lower-dimensional tensors (for axis reductions).

---

## Responsibilities

The reductions module is responsible for:

- Computing global scalar reductions (sum, product, max, min, mean, variance, std deviation)
- Computing indices of extrema (argmax, argmin)
- Reducing along specified axes while preserving other dimensions
- Type-safe accumulation with proper initializers

The reductions module is **not** responsible for:

- Non-owning reduction views
- Custom reduction kernels
- Weighted reductions

---

## Relationships

```text
reduction::* functions
├── sum, prod, max, min
├── argmax, argmin
├── mean, var, std
└── axis_reduce(...) generic framework
```

Depends on:

- `include/stratax/core/dtypes/Concepts.hpp` (Array concept)
- `include/stratax/containers/Tensor.hpp`
- `include/stratax/indexing/Slicing.hpp`
- `<numeric>`, `<algorithm>`, `<cmath>`

Used by:

- High-level tensor algorithms
- Statistical operations
- Model training/optimization

---

## Public Interface

### Global Reductions

All global reductions operate on the entire container and return a scalar value.

#### sum

```cpp
template<Array A>
typename A::value_type sum(const A& arr);
```

Returns the sum of all elements.

Throws

- None (uses `std::accumulate`)

Complexity

- O(n)

#### prod

```cpp
template<Array A>
typename A::value_type prod(const A& arr);
```

Returns the product of all elements.

Complexity

- O(n)

#### max / min

```cpp
template<Array A>
typename A::value_type max(const A& arr);

template<Array A>
typename A::value_type min(const A& arr);
```

Returns the maximum or minimum element.

Complexity

- O(n)

#### argmax / argmin

```cpp
template<Array A>
std::size_t argmax(const A& arr);

template<Array A>
std::size_t argmin(const A& arr);
```

Returns the flat index of the maximum or minimum element.

Complexity

- O(n)

#### mean

```cpp
template<Array A>
double mean(const A& arr);
```

Returns the arithmetic mean of all elements.

Complexity

- O(n)

#### var / std

```cpp
template<Array A>
double var(const A& arr);

template<Array A>
double std(const A& arr);
```

Returns the variance or standard deviation of all elements.

Complexity

- O(n)

---

### Axis Reductions

Axis reductions reduce along a specified dimension while preserving all other dimensions.

Axis normalization

- Positive axes use the standard zero-based indexing (`0 .. rank-1`).
- Negative axes are normalized from the end (`-1` is the last axis, `-2` is the second-to-last axis, and so on).
- Valid axis range is `[-rank, rank-1]`.
- An out-of-range axis throws `Exceptions::AxisError`.

#### sum(arr, axis)

```cpp
template<Array A>
stratax::container::Tensor<typename A::value_type> sum(const A& arr, int axis);
```

Returns a tensor with dimension `axis` removed, containing element-wise sums along that axis.

Result shape

- Original shape with dimension `axis` removed

Complexity

- O(n * rank)

Example

```cpp
Matrix<int> m{{1, 2, 3}, {4, 5, 6}};
auto col_sums = reduction::sum(m, 0);  // [5, 7, 9]
auto row_sums = reduction::sum(m, 1);  // [6, 15]
auto row_sums_neg = reduction::sum(m, -1);  // [6, 15]
```

#### prod(arr, axis) / max(arr, axis) / min(arr, axis)

```cpp
template<Array A>
stratax::container::Tensor<typename A::value_type> prod(const A& arr, int axis);

template<Array A>
stratax::container::Tensor<typename A::value_type> max(const A& arr, int axis);

template<Array A>
stratax::container::Tensor<typename A::value_type> min(const A& arr, int axis);
```

Same semantics as `sum(arr, axis)` but return products, maxima, or minima.

#### argmax(arr, axis) / argmin(arr, axis)

```cpp
template<Array A>
stratax::container::Tensor<std::size_t> argmax(const A& arr, int axis);

template<Array A>
stratax::container::Tensor<std::size_t> argmin(const A& arr, int axis);
```

Returns a tensor of **flat indices** (within the sliced dimension) of the maximum or minimum elements along each reduction.

Result shape

- Original shape with dimension `axis` removed

Example

```cpp
Matrix<int> m{{3, 1, 4}, {2, 7, 1}};
auto max_indices = reduction::argmax(m, 1);  // [2, 1]
auto min_indices = reduction::argmin(m, 1);  // [1, 2]
```

#### mean(arr, axis) / var(arr, axis) / std(arr, axis)

```cpp
template<Array A>
stratax::container::Tensor<double> mean(const A& arr, int axis);

template<Array A>
stratax::container::Tensor<double> var(const A& arr, int axis);

template<Array A>
stratax::container::Tensor<double> std(const A& arr, int axis);
```

Compute mean, variance, and standard deviation along the specified axis.

Result type

- Always `Tensor<double>` for numerical precision

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Global reductions (sum, prod, max, min, mean) | O(n) |
| argmax / argmin | O(n) |
| Axis reductions | O(n * rank) |
| var / std | O(n) |

---

## Examples

### Global Reduction Examples

```cpp
Vector<int> v{1, 2, 3, 4, 5};

int total = reduction::sum(v);     // 15
int product = reduction::prod(v);  // 120
int maximum = reduction::max(v);   // 5
int minimum = reduction::min(v);   // 1

std::size_t max_idx = reduction::argmax(v);  // 4
std::size_t min_idx = reduction::argmin(v);  // 0

double average = reduction::mean(v);  // 3.0
```

### Matrix Axis Reductions

```cpp
Matrix<int> m{
    {1, 2, 3},
    {4, 5, 6}
};

// Reduce along axis 0 (sum columns)
auto col_sums = reduction::sum(m, 0);    // [5, 7, 9]
auto col_maxes = reduction::max(m, 0);   // [4, 5, 6]

// Reduce along axis 1 (sum rows)
auto row_sums = reduction::sum(m, 1);    // [6, 15]
auto row_mins = reduction::min(m, 1);    // [1, 4]
```

### 3D Tensor Axis Reductions

```cpp
Tensor<int> t(Shape{2, 3, 4});
// ... fill tensor ...

// Reduce first dimension
auto t_axis0 = reduction::sum(t, 0);  // Shape [3, 4]

// Reduce second dimension
auto t_axis1 = reduction::sum(t, 1);  // Shape [2, 4]

// Negative axis resolves from the end
auto t_last_axis = reduction::sum(t, -1);  // Shape [2, 3]

// Find max indices along last dimension
auto max_idx = reduction::argmax(t, 2);  // Shape [2, 3] of size_t
```

---

## Design Notes

- **Type Safety**: The `axis_reduce` template accepts any callable, enabling custom operations beyond built-in reductions.
- **Value Semantics**: All axis reductions return owning tensors, not views.
- **Numerical Precision**: Mean, variance, and std deviation return `double` for stability.
- **Row-Major Iteration**: The `advance()` utility handles multi-dimensional iteration without nested loops.

---

## Future Improvements

- Add weighted reductions
- Add cumulative reductions (cumsum, cumprod)
- Add reductions along multiple axes simultaneously
- Add numerically stable variance/std algorithms

---

## See Also

- `include/stratax/algorithms/Creation.hpp`
- `include/stratax/algorithms/Reshape.hpp`
- `include/stratax/containers/Tensor.hpp`
