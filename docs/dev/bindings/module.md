@page dev_module "_core Extension Module"

# _core Extension Module

Version: v0.2.0

Status: Complete

Source: `bindings/module.cpp`

---

## Overview

`bindings/module.cpp` defines the pybind11 extension module `_core` and orchestrates all binding registration calls.

It is the compiled binding entry point re-exported by the public `stratax` package.

---

## Responsibilities

The module entry file is responsible for:

- Declaring `PYBIND11_MODULE(_core, m)`
- Setting module metadata (`__doc__`, `__version__`, `__author__`)
- Registering Stratax exception types
- Registering the `Shape` metadata type
- Calling all bind registration functions in a consistent order

The module entry file is not responsible for:

- Implementing vector, matrix, tensor, algorithm, or creation binding logic directly
- Defining the public package export list (handled in `python/stratax/__init__.py`)

---

## Registration Order

`bind_*` calls currently execute in this order:

1. `bind_exceptions`
2. `bind_shape`
3. `bind_vector`
4. `bind_matrix`
5. `bind_tensor`
6. `bind_conversions`
7. `bind_creation`
8. `bind_reductions`

This order ensures exceptions and `Shape` are registered before container and helper registrations.

---

## Metadata Sources

Module metadata values are sourced from `include/stratax/core/Meta.hpp`:

- `STRATAX_PY_DOC` -> `m.attr("__doc__")`
- `STRATAX_VERSION` -> `m.attr("__version__")`
- `STRATAX_AUTHOR` -> `m.attr("__author__")`

---

## Related Files

- `bindings/utils.hpp`
- `bindings/vector.cpp`
- `bindings/matrix.cpp`
- `bindings/tensor.cpp`
- `bindings/conversions.cpp`
- `bindings/creation.cpp`
- `bindings/reductions.cpp`

---

## Future Improvements

- Add explicit registration comments for dependency expectations
- Add startup smoke test asserting required symbols are exported
