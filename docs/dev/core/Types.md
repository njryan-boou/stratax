# Types

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/Types.hpp`

---

## Overview

`Types.hpp` provides canonical scalar and index aliases used across Stratax core APIs.

These aliases map directly to standard C++ types and `std::complex` specializations.

---

## Responsibilities

The types module is responsible for:

- Defining stable alias names for supported scalar categories
- Defining index aliases for size and signed-size usage
- Serving as a common type vocabulary for concepts, containers, and bindings

The types module is not responsible for:

- Runtime dtype dispatch or metadata
- Type acceptance rules for algorithms or containers
- Numeric promotion policy

---

## Relationships

```text
stratax::core::dtype
├── index aliases (index_t, ssize_t)
├── boolean/integer aliases
├── floating aliases
└── complex aliases
```

Depends on:

- C++ headers: `<cstddef>`, `<cstdint>`, `<complex>`

Used by:

- `include/stratax/core/Concepts.hpp`
- Container templates and algorithm signatures

---

## Invariants

The following conditions are always true:

- Aliases are compile-time names only and add no runtime state.
- All aliases map to standard library/fundamental C++ types.
- Alias stability is expected across core APIs unless versioned changes are introduced.

---

## Public Interface

Namespace:

```cpp
namespace stratax::core::dtype { ... }
```

### Index aliases

```cpp
using index_t = std::size_t;
using ssize_t = std::ptrdiff_t;
```

### Boolean alias

```cpp
using bool_ = bool;
```

### Integer aliases

```cpp
using int8  = std::int8_t;
using int16 = std::int16_t;
using int32 = std::int32_t;
using int64 = std::int64_t;

using uint8  = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;
using uint64 = std::uint64_t;
```

### Floating aliases

```cpp
using float32  = float;
using float64  = double;
using float128 = long double;
```

### Complex aliases

```cpp
using complex64  = std::complex<float>;
using complex128 = std::complex<double>;
using complex256 = std::complex<long double>;
```

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Alias usage and substitution | Compile-time only |
| Runtime overhead introduced by aliases | O(0) |

---

## Examples

```cpp
using stratax::core::dtype::float64;
using stratax::core::dtype::index_t;

float64 value = 3.14159;
index_t n = 128;
```

---

## Design Notes

Aliases keep public APIs readable and consistent across C++ and bindings-facing layers.

Eligibility for concepts like `Numeric` is determined in `Concepts.hpp`, not in this file.

---

## Future Improvements

- Add explicit dtype traits and canonical name helpers
- Add documented promotion policies across alias categories
- Revisit `float128` portability guarantees across compilers/platforms

---

## See Also

- `include/stratax/core/Concepts.hpp`
