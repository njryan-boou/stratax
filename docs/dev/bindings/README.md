# Bindings Developer Docs

This section documents the pybind11 binding layer under `bindings/`.

---

## Overview

Stratax Python bindings are compiled into the `_core` extension module and then surfaced by the Python package wrappers under `python/stratax/`.

Current bindings are focused on `double`-based container types and free-function helpers.

---

## Entry Points

- [Extension Module (`_core`)](module.md)

---

## File Map

- `bindings/module.cpp`: extension module definition, metadata wiring, and bind-order orchestration
- `bindings/common/`: shared exception translation, helper functions, and utility adapters
- `bindings/shape/`: `Shape` Python API binding surface
- `bindings/vector/`: `Vector<double>` bindings (constructors, indexing, properties, ops, reshape)
- `bindings/matrix/`: `Matrix<double>` bindings (constructors, indexing, properties, ops, reshape)
- `bindings/tensor/`: `Tensor<double>` bindings (constructors, indexing, properties, ops, reshape)
- `bindings/conversions.cpp`: free-function conversions (`to_vector`, `to_matrix`, `to_tensor`)
- `bindings/creation.cpp`: free-function creation helpers (`zeros`, `ones`, `full`, `identity`)

---

## Data Flow

```text
C++ core headers -> pybind11 bindings in bindings/ -> _core extension
-> python/stratax wrapper modules -> user-facing Python API
```

---

## Current Constraints

- Numeric bindings currently expose `double` container specializations.
- Conversion and creation helpers in Python are module-level free functions.
- Extension metadata is sourced from `include/stratax/core/Meta.hpp`.

---

## Future Improvements

- Extend bindings to additional numeric dtypes
- Add explicit binding docs per container subfolder
- Add binding-level exception mapping matrix
