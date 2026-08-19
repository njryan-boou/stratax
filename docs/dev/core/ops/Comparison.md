@page comparison Comparison

# Comparison {#dev_comparison}

Version: v0.2.0

Status: Complete

Header: `include/stratax/ops/Comparison.hpp`

---

## Overview

`Comparison.hpp` provides element-wise equality and ordering comparisons for
Stratax arrays. Every comparison returns a newly allocated array whose dtype is
`stratax::dtype::bool_`; it does not reduce the result to one Boolean value.

The API supports all operand arrangements:

- Array compared with array
- Array compared with scalar
- Scalar compared with array

Array-array comparisons follow Stratax broadcasting rules and may mix
`Vector`, `Matrix`, and `Tensor` operands. Array-scalar comparisons preserve the
array operand's container type and shape.

---

## Responsibilities

The comparison module is responsible for:

- Applying exact element-wise equality and inequality
- Applying element-wise ordering comparisons to ordered dtypes
- Broadcasting compatible array-array operands
- Preserving operand order in scalar-array comparisons
- Returning owning Boolean arrays without modifying either operand
- Promoting mixed array container types to the appropriate result container

The comparison module is not responsible for:

- Reducing a Boolean array to a single `bool`
- Approximate or tolerance-based floating-point equality
- Comparing incompatible array shapes
- Ordering complex values

---

## Relationships

```text
Public named functions and operators
|-- comparison_detail::comparison_op(...)
|   |-- broadcasted_shape(...)
|   `-- broadcast_detail::flat_operand_index(...)
|-- comparison_detail::comparison_scalar_op(...)
|-- core::promote_array_t<..., dtype::bool_>
`-- core::rebind_array_t<..., dtype::bool_>
```

Depends on:

- `include/stratax/core/dtypes/Concepts.hpp`
- `include/stratax/core/dtypes/ArrayTraits.hpp`
- `include/stratax/core/dtypes/Types.hpp`
- `include/stratax/ops/Broadcasting.hpp`

---

## Invariants

The following conditions are always true:

- Every public comparison returns an owning Boolean array.
- Array-array results use the common broadcasted shape.
- Mixed array container types use `promote_array_t` to select the result
  container.
- Array-scalar results preserve the array operand's shape and container type.
- Scalar-array operations preserve left-to-right operand order.
- Comparisons do not mutate their operands.
- Equality is exact and uses the underlying dtype's `==` operation.
- Ordering operations require both participating dtypes to satisfy `Ordered`.

---

## Public Interface

### Named array-array comparisons

```cpp
template<Array L, Array R> auto equal(const L& lhs, const R& rhs);
template<Array L, Array R> auto not_equal(const L& lhs, const R& rhs);

template<Array L, Array R>
requires Ordered<typename L::value_type> &&
         Ordered<typename R::value_type>
auto less(const L& lhs, const R& rhs);

template<Array L, Array R>
requires Ordered<typename L::value_type> &&
         Ordered<typename R::value_type>
auto less_equal(const L& lhs, const R& rhs);

template<Array L, Array R>
requires Ordered<typename L::value_type> &&
         Ordered<typename R::value_type>
auto greater(const L& lhs, const R& rhs);

template<Array L, Array R>
requires Ordered<typename L::value_type> &&
         Ordered<typename R::value_type>
auto greater_equal(const L& lhs, const R& rhs);
```

The result container is `stratax::core::promote_array_t<L, R,
stratax::dtype::bool_>`. Its shape is the common broadcasted shape.

Throws

- `Exceptions::BroadcastError` when the operand shapes are incompatible

Complexity

- O(nr), where `n` is the result element count and `r` is the result rank

### Named array-scalar comparisons

Equality and inequality accept any `DType` scalar:

```cpp
template<Array A, DType Scalar> auto equal(const A& lhs, const Scalar& rhs);
template<DType Scalar, Array A> auto equal(const Scalar& lhs, const A& rhs);

template<Array A, DType Scalar> auto not_equal(const A& lhs, const Scalar& rhs);
template<DType Scalar, Array A> auto not_equal(const Scalar& lhs, const A& rhs);
```

Ordering overloads use the same operand arrangements but require the scalar and
array value type to satisfy `Ordered`:

```cpp
less(array, scalar);          less(scalar, array);
less_equal(array, scalar);    less_equal(scalar, array);
greater(array, scalar);       greater(scalar, array);
greater_equal(array, scalar); greater_equal(scalar, array);
```

The result type is `stratax::core::rebind_array_t<A,
stratax::dtype::bool_>`. No broadcasting is needed because the scalar is
applied to every array element.

Complexity

- O(n), where `n` is the array element count

### Operators

All named comparisons have equivalent operator syntax for array-array,
array-scalar, and scalar-array operands:

```cpp
lhs == rhs;
lhs != rhs;
lhs < rhs;
lhs <= rhs;
lhs > rhs;
lhs >= rhs;
```

The operators return the same Boolean array type as their corresponding named
function. In particular, `lhs == rhs` is an element-wise operation and cannot
be used directly as a single Boolean condition.

---

## Internal Helpers

Implementation helpers live in `stratax::core::comparison_detail`:

- `comparison_op` performs broadcasted array-array traversal.
- `comparison_scalar_op` implements both scalar operand orders.
- `array_equal` performs internal whole-array shape and value equality.

These helpers are implementation details and are not part of the public API.

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Array-array comparison | O(nr) |
| Array-scalar comparison | O(n) |
| Scalar-array comparison | O(n) |
| Internal whole-array equality | O(r + n) worst case |

For broadcasted operations, `n` is the result size and `r` is the result rank.
For scalar operations, `n` is the array size.

---

## Examples

### Array-array broadcasting

```cpp
const Matrix<int> column{{1}, {3}};
const Matrix<int> row{{2, 3, 4}};

const auto mask = column < row;
// shape: (2, 3)
// values: {{true, true, true}, {false, false, true}}
```

### Scalar comparisons

```cpp
const Vector<int> values{1, 2, 3};

const auto at_least_two = values >= 2;
// {false, true, true}

const auto two_is_greater = 2 > values;
// {true, false, false}
```

### Mixed containers and dtypes

```cpp
const Vector<stratax::dtype::int16> vector{1, 2, 3};
const Matrix<stratax::dtype::float32> matrix{{1.0F}, {2.5F}};

const auto mask = vector < matrix;
// Result type: Tensor<stratax::dtype::bool_>
// Result shape: (2, 3)
```

### Whole-result checks

Because comparisons return arrays, callers that need one Boolean value should
explicitly reduce or inspect the mask rather than treating it as a `bool`:

```cpp
const auto mask = lhs == rhs;
const bool first_elements_match = mask[0];
```

---

## Design Notes

Equality remains exact for floating-point and complex values. Approximate
comparison should be exposed as a separate, explicit algorithm so tolerance
selection is visible at the call site.

Complex dtypes support `equal`, `not_equal`, `==`, and `!=`. They do not satisfy
`Ordered`, so `<`, `<=`, `>`, and `>=` are rejected at compile time.

---

## Future Improvements

- Add explicit tolerance-based floating-point comparison algorithms
- Add public `all` and `any` Boolean reductions for whole-mask queries

---

## See Also

- `include/stratax/ops/Broadcasting.hpp`
- `include/stratax/ops/Arithmetic.hpp`
- `include/stratax/core/dtypes/Concepts.hpp`
- `include/stratax/core/dtypes/ArrayTraits.hpp`
