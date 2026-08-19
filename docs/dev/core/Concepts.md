@page concepts Concepts

# Numeric, DType, and Array Concepts {#dev_concepts}

Version: v0.2.0

Status: Complete

Header: `include/stratax/concepts/Numeric.hpp`

---

## Overview

`Numeric.hpp` defines the compile-time type vocabulary used by Stratax
containers and operations:

- `Integral<T>` accepts integer scalars while excluding boolean and character
  types.
- `Numeric<T>` accepts supported integral, floating-point, and complex
  scalars.
- `DType<T>` accepts every `Numeric` type plus `bool` for array storage.
- `is_array<T>` identifies exact owning Stratax container specializations.
- `Array<T>` applies cv/ref normalization before using `is_array`.

All checks happen at compile time and add no runtime work.

---

## Public Interface

```cpp
template<typename T>
concept Integral;

template<typename T>
concept Numeric;

template<typename T>
concept DType;

template<typename T>
struct is_array;

template<typename T>
concept Array;
```

The concepts and trait are declared in the global namespace. Container forward
declarations are provided in `stratax::container`.

---

## Internal Classification Concepts

The `stratax::core::concept_detail` namespace contains implementation helpers:

```cpp
template<typename T, typename... Candidates>
concept SameAsAny;

template<typename T>
concept CharacterLike;

template<typename T>
concept BoolLike;

template<typename T>
concept SupportedComplex;
```

`SameAsAny` removes cv- and reference qualifiers from `T`, then checks for an
exact match with any candidate type.

`CharacterLike` recognizes `char`, `signed char`, `unsigned char`, `wchar_t`,
`char8_t`, `char16_t`, and `char32_t`. These are excluded from numerical
storage even though the standard library classifies them as integral.

`BoolLike` recognizes `bool` after removing cv/ref qualifiers.

`SupportedComplex` recognizes exactly:

- `std::complex<float>`
- `std::complex<double>`
- `std::complex<long double>`

Other `std::complex<T>` specializations do not satisfy `Numeric`.

---

## Integral

```cpp
template<typename T>
concept Integral =
    std::integral<std::remove_cvref_t<T>> &&
    !stratax::core::concept_detail::BoolLike<T> &&
    !stratax::core::concept_detail::CharacterLike<T>;
```

`Integral` ignores cv/ref qualifiers and accepts non-character signed and
unsigned integer types.

| Type | `Integral` |
| ---- | :--------: |
| `short`, `int`, `long`, `long long` | Yes |
| Unsigned integer counterparts | Yes |
| `const int&` | Yes |
| `bool` | No |
| Character and code-unit types | No |
| Floating-point types | No |
| Enumeration types | No |

```cpp
static_assert(Integral<int>);
static_assert(Integral<const unsigned long&>);
static_assert(!Integral<bool>);
static_assert(!Integral<char>);
```

---

## Numeric

```cpp
template<typename T>
concept Numeric =
    Integral<T> ||
    std::floating_point<std::remove_cvref_t<T>> ||
    stratax::core::concept_detail::SupportedComplex<T>;
```

`Numeric` is the scalar constraint used by numerical algorithms and operators.
Array containers use the broader `DType` concept described below.

| Category | Accepted types |
| -------- | -------------- |
| Integer | Types satisfying `Integral` |
| Floating point | `float`, `double`, `long double` |
| Complex | `std::complex<float|double|long double>` |

Rejected categories include booleans, character types, strings, pointers,
enumerations, and arbitrary user-defined arithmetic-like classes.

```cpp
static_assert(Numeric<int>);
static_assert(Numeric<const double&>);
static_assert(Numeric<std::complex<float>>);
static_assert(!Numeric<bool>);
static_assert(!Numeric<char16_t>);
static_assert(!Numeric<std::complex<int>>);
```

The concept performs classification only. It does not define conversion,
promotion, precision, overflow, or runtime validation policy.

---

## DType

```cpp
template<typename T>
concept DType =
    Numeric<T> ||
    stratax::core::concept_detail::BoolLike<T>;
```

`DType` defines the element types accepted by `Vector<T>`, `Matrix<T>`, and
`Tensor<T>`. It extends `Numeric` with `bool`, allowing logical arrays without
classifying booleans as integers or enabling them for arithmetic-only APIs.
Like the other concepts, it ignores cv/ref qualifiers.

| Category | `DType` |
| -------- | :-----: |
| Types satisfying `Numeric` | Yes |
| `bool` | Yes |
| Character and code-unit types | No |
| Strings, pointers, and arbitrary classes | No |

```cpp
static_assert(DType<bool>);
static_assert(DType<const bool&>);
static_assert(DType<double>);
static_assert(!DType<char>);
```

---

## Container Forward Declarations

```cpp
namespace stratax::container {

template<typename T>
requires DType<T>
class Vector;

template<typename T>
requires DType<T>
class Matrix;

template<typename T>
requires DType<T>
class Tensor;

}
```

These declarations allow the array trait specializations to be defined without
including the full container implementations.

---

## is_array Trait

```cpp
template<typename T>
struct is_array : std::false_type {};
```

The trait is specialized to derive from `std::true_type` for:

- `stratax::container::Vector<T>`
- `stratax::container::Matrix<T>`
- `stratax::container::Tensor<T>`

where `T` satisfies `DType`.

`is_array` checks its argument exactly and does not remove qualifiers:

```cpp
using Vector = stratax::container::Vector<int>;

static_assert(is_array<Vector>::value);
static_assert(!is_array<const Vector>::value);
```

Use the `Array` concept when cv/ref normalization is required.

---

## Array

```cpp
template<typename T>
concept Array =
    is_array<std::remove_cvref_t<T>>::value;
```

`Array` recognizes the three supported owning Stratax containers and ignores
cv/ref qualifiers:

```cpp
using Vector = stratax::container::Vector<int>;

static_assert(Array<Vector>);
static_assert(Array<const Vector&>);
static_assert(!Array<int>);
static_assert(!Array<std::vector<int>>);
```

This is a nominal concept, not a structural one. A third-party type does not
satisfy `Array` merely because it provides `shape()`, `size()`, or iterators.

---

## Usage

```cpp
template<Numeric T>
T square(T value)
{
    return value * value;
}

template<Array A>
void clear(A& array)
{
    array.fill(typename A::value_type{});
}
```

Constraints reject unsupported types during template substitution rather than
performing runtime checks.

---

## Complexity

| Operation | Runtime complexity |
| --------- | -----------------: |
| `Integral<T>` evaluation | O(0) |
| `Numeric<T>` evaluation | O(0) |
| `DType<T>` evaluation | O(0) |
| `is_array<T>` evaluation | O(0) |
| `Array<T>` evaluation | O(0) |

All results are compile-time constants.

---

## Design Notes

Character exclusions prevent accidental use of textual data as array storage.
Boolean values are accepted through `DType` for logical arrays but remain
excluded from `Numeric`, `Integral`, and arithmetic-only APIs. The explicit
complex whitelist keeps supported representations aligned with standard
floating-point precision types.

`Array` deliberately recognizes only Stratax owning containers. This gives
operators and algorithms a precise boundary and avoids accepting unrelated
types based on coincidental member names.

---

## Future Improvements

- Add focused concepts when rank-specific constraints become useful
- Add an opt-in extension mechanism for external array types if required
- Define promotion-oriented concepts alongside a mixed-type promotion policy

---

## See Also

- @ref vector
- @ref matrix
- @ref tensor
- `include/stratax/core/validation/TypeValidation.hpp`
