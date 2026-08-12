@page dev_bindings Bindings Developer Docs

# Bindings Developer Docs

This section documents the pybind11 binding layer under `bindings/`.

---

## Overview

Stratax Python bindings are compiled into the `_core` extension module and re-exported by the `stratax` Python package.

Current bindings are focused on `double`-based container types and free-function helpers.

---

## Entry Points

- @ref dev_module "Extension Module (`_core`)"

---

## File Map

- `bindings/module.cpp`: extension module definition, metadata wiring, and bind-order orchestration
- `bindings/shape.cpp`: `Shape` metadata bindings (constructors, indexing, properties, iteration)
- `bindings/utils.hpp`: shared conversion, overflow, index, and slice helpers
- `bindings/vector.cpp`: `Vector<double>` bindings (constructors, indexing, properties, ops, reshape)
- `bindings/matrix.cpp`: `Matrix<double>` bindings (constructors, indexing, properties, ops, reshape)
- `bindings/tensor.cpp`: `Tensor<double>` bindings (constructors, indexing, properties, ops, reshape)
- `bindings/conversions.cpp`: free-function conversions (`to_vector`, `to_matrix`, `to_tensor`)
- `bindings/creation.cpp`: free-function creation helpers (`zeros`, `ones`, `full`, `identity`)
- `bindings/reductions.cpp`: free-function reductions (`sum`, `prod`, `max`, `min`, `argmax`, `argmin`, `mean`, `var`, `std`)

---

## Data Flow

```text
C++ core headers -> pybind11 bindings in bindings/ -> _core extension
-> python/stratax package exports -> user-facing Python API
```

---

## Current Constraints

- Numeric bindings currently expose `double` container specializations.
- Conversion and creation helpers in Python are module-level free functions.
- Extension metadata is defined directly in `bindings/module.cpp`.

---

## Future Improvements

- Extend bindings to additional numeric dtypes
- Add binding-level smoke tests for import and symbol export
- Add binding-level exception mapping matrix
