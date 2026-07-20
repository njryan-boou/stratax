# Config {#dev_config}

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/Config.hpp`

---

## Overview

`Config.hpp` defines compile-time configuration constants for core runtime behavior.

These values are exposed as inline `constexpr` constants under `stratax::core::config`.

---

## Responsibilities

The config module is responsible for:

- Providing default alignment configuration
- Declaring build-time feature toggles for bounds checking and acceleration paths
- Supplying simple globally visible constants used by core code

The config module is not responsible for:

- Runtime configuration mutation
- Environment-variable or file-based settings
- Backend implementation details for SIMD, threading, or CUDA

---

## Relationships

```text
stratax::core::config
|- default_alignment
|- bounds_checking
|- use_simd
|- use_multithreading
`- use_cuda
```

Depends on:

- C++ header `<cstddef>`

Used by:

- Core allocation and execution-path selection logic

---

## Invariants

The following conditions are always true:

- All configuration values are compile-time constants.
- Values are immutable during runtime.
- Feature toggles express whether a code path may be used, not whether it is currently active.

---

## Public Interface

Namespace:

```cpp
namespace stratax::core::config { ... }
```

Constants:

```cpp
inline constexpr std::size_t default_alignment = 64;
inline constexpr bool bounds_checking = true;
inline constexpr bool use_simd = true;
inline constexpr bool use_multithreading = false;
inline constexpr bool use_cuda = false;
```

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Accessing config constants | O(1) |
| Runtime overhead from constants | O(0) |

---

## Examples

```cpp
if constexpr (stratax::core::config::bounds_checking)
{
    // bounds validation path
}

constexpr auto alignment = stratax::core::config::default_alignment;
```

---

## Design Notes

Keeping configuration as inline `constexpr` values makes behavior explicit and compiler-friendly across translation units.

---

## Future Improvements

- Add documented build-profile presets for debug/performance modes
- Introduce explicit configuration layering if runtime overrides are ever required

---

## See Also

- `include/stratax/core/Meta.hpp`
