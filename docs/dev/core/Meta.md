# Meta

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/Meta.hpp`

---

## Overview

`Meta.hpp` defines preprocessor metadata macros describing the Stratax project and Python binding surface.

The file currently exposes version, bindings description, and author metadata.

---

## Responsibilities

The meta module is responsible for:

- Publishing project metadata at preprocess time
- Providing a canonical version string macro
- Providing basic author and bindings description macros

The meta module is not responsible for:

- Semantic version parsing logic
- Runtime metadata registries
- Build-time compatibility checks

---

## Relationships

```text
Meta.hpp
|- STRATAX_VERSION
|- STRATAX_PY_DOC
`- STRATAX_AUTHOR
```

Used by:

- Build/banner output
- Binding metadata exposure
- Documentation/version stamping

---

## Invariants

The following conditions are always true:

- Metadata values are macro string literals.
- Macros are available after including the header.
- No runtime storage or initialization is required.

---

## Public Interface

```cpp
#define STRATAX_VERSION "0.2.0"
#define STRATAX_PY_DOC "Python bindings for Stratax"
#define STRATAX_AUTHOR "Noah Ryan"
```

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Macro expansion | Compile-time only |
| Runtime overhead | O(0) |

---

## Examples

```cpp
#include <stratax/core/Meta.hpp>

static_assert(sizeof(STRATAX_VERSION) > 1);
```

---

## Design Notes

The metadata is intentionally minimal and macro-based for easy reuse in C++ and binding generation contexts.

---

## Future Improvements

- Add machine-readable version components (major/minor/patch)
- Consider constexpr alternatives alongside macros

---

## See Also

- `include/stratax/core/Config.hpp`
