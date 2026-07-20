# Concepts {#dev_concepts}

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/Concepts.hpp`

---

## Overview

`Concepts.hpp` defines Stratax compile-time constraints for scalar categories and array-like container categories.

These concepts gate template overloads and class declarations without adding runtime overhead.

---

## Responsibilities

The concepts module is responsible for:

- Classifying scalar types into numeric categories
- Restricting accepted array container types for generic ops
- Providing structural ND-array concept checks for generic algorithms

The concepts module is not responsible for:

- Runtime validation or exception throwing
- Value-level semantics (broadcasting, promotion, precision policy)
- Container implementation details

---

## Relationships

```text
concept_detail
├── clean_t
├── SameAsAny
├── CharacterLike / BoolLike
└── SupportedComplex

Public concepts
├── Integral / Integer / SignedInteger / UnsignedInteger
├── Floating / Float / Arithmetic / Real
├── Complex / Numeric / Scalar
└── Array / NDarray
```

Depends on:

- `include/stratax/core/Types.hpp`
- C++ headers: `<concepts>`, `<complex>`, `<type_traits>`

Used by:

- Container templates (`Vector`, `Matrix`, `Tensor`)
- Ops and algorithms overload constraints

---

## Invariants

The following conditions are always true:

- Type checks normalize cv/ref qualifiers via `clean_t`.
- `Integral` excludes bool-like and character-like types.
- `Complex` accepts only supported `std::complex<float|double|long double>`.
- `Numeric` equals `Arithmetic || Complex`.
- `Array` recognizes Stratax container specializations via `is_array`.
- `NDarray` is structural (`shape`, `size`, `begin`, `end` required).

---

## Public Interface

### Scalar concepts

```cpp
template<typename T> concept Integral;
template<typename T> concept Integer;
template<typename T> concept SignedInteger;
template<typename T> concept UnsignedInteger;

template<typename T> concept Floating;
template<typename T> concept Float;
template<typename T> concept Arithmetic;
template<typename T> concept Real;

template<typename T> concept Complex;
template<typename T> concept Numeric;
template<typename T> concept Scalar;
```

### Container declarations and traits

```cpp
namespace stratax::container {
template<typename T> requires Numeric<T> class Vector;
template<typename T> requires Numeric<T> class Matrix;
template<typename T> requires Numeric<T> class Tensor;
}

template<typename T> struct is_array;
```

### Container concepts

```cpp
template<typename T> concept Array;
template<typename T> concept NDarray;
```

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Concept checks (`Integral`, `Numeric`, `Array`) | Compile-time only |
| Structural requirement checks (`NDarray`) | Compile-time only |
| Runtime overhead introduced by concepts | O(0) |

---

## Examples

```cpp
template<Numeric T>
T twice(T value)
{
    return value + value;
}

template<Array A>
A add_arrays(const A& lhs, const A& rhs)
{
    return lhs + rhs;
}
```

---

## Design Notes

Concepts deliberately separate accepted Stratax array types (`Array`) from structural array-like requirements (`NDarray`) so APIs can choose strict or generic constraints.

Character-like exclusions prevent accidental acceptance of textual byte-like types as numeric scalar data.

---

## Future Improvements

- Add concepts for matrix-only and tensor-only constraints
- Add explicit promotion-related concepts for mixed-type arithmetic
- Add regression tests for accepted/rejected edge-case types

---

## See Also

- `include/stratax/core/Types.hpp`
- `include/stratax/core/validation/TypeValidation.hpp`
