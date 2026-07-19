# _core Extension Module

Version: v0.2.0

Status: Complete

Source: `bindings/module.cpp`

---

## Overview

`bindings/module.cpp` defines the pybind11 extension module `_core` and orchestrates all binding registration calls.

It is the top-level binding entry point loaded by the Python wrapper package.

---

## Responsibilities

The module entry file is responsible for:

- Declaring `PYBIND11_MODULE(_core, m)`
- Setting module metadata (`__doc__`, `__version__`, `__author__`)
- Calling all bind registration functions in a consistent order

The module entry file is not responsible for:

- Implementing per-type binding logic directly
- Defining Python wrapper ergonomics (handled in `python/stratax/`)

---

## Registration Order

`bind_*` calls currently execute in this order:

1. `bind_common_exceptions`
2. `bind_common_helpers`
3. `bind_common_utilities`
4. `bind_shape`
5. `bind_vector`
6. `bind_matrix`
7. `bind_tensor`
8. `bind_conversions`
9. `bind_creation`

This order ensures shared infrastructure is available before container and helper registrations.

---

## Metadata Sources

Module metadata values are sourced from `include/stratax/core/Meta.hpp`:

- `STRATAX_PY_DOC` -> `m.attr("__doc__")`
- `STRATAX_VERSION` -> `m.attr("__version__")`
- `STRATAX_AUTHOR` -> `m.attr("__author__")`

---

## Related Files

- `bindings/common/exceptions.cpp`
- `bindings/common/helpers.cpp`
- `bindings/common/utilities.cpp`
- `bindings/shape/bind.cpp`
- `bindings/vector/bind.cpp`
- `bindings/matrix/bind.cpp`
- `bindings/tensor/bind.cpp`
- `bindings/conversions.cpp`
- `bindings/creation.cpp`

---

## Future Improvements

- Add explicit registration comments for dependency expectations
- Add startup smoke test asserting required symbols are exported
