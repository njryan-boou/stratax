# Exceptions

Version: v0.2.0

Status: Complete

Header: `include/stratax/core/Exceptions.hpp`

---

## Overview

`Exceptions.hpp` defines Stratax's domain-specific runtime error hierarchy.

All Stratax exceptions derive from `std::runtime_error` through `Exceptions::StrataxError` and are used to communicate shape, dimension, indexing, typing, broadcasting, and arithmetic-domain failures.

---

## Responsibilities

The exception module is responsible for:

- Providing a common base type for Stratax runtime failures
- Defining specific exception categories for common error domains
- Carrying human-readable error messages via standard runtime-error constructors

The exception module is **not** responsible for:

- Performing validation itself
- Defining recovery policy
- Implementing logging or diagnostics transport

---

## Relationships

```text
std::runtime_error
└── Exceptions::StrataxError
    ├── Exceptions::ShapeError
    ├── Exceptions::DimensionError
    ├── Exceptions::IndexError
    ├── Exceptions::TypeError
    ├── Exceptions::BroadcastError
    └── Exceptions::ZeroDivisionError
```

Depends on:

- C++ standard library `<stdexcept>`

Used by:

- Container classes (`Buffer`, `Shape`, `Strides`, `Vector`, `Matrix`, `Tensor`)
- Validation helpers (`validation/*`)
- Operation helpers (`ops/*`, `algorithms/*`)

---

## Invariants

The following conditions are always true:

- Every Stratax-specific exception derives from `Exceptions::StrataxError`.
- `Exceptions::StrataxError` derives from `std::runtime_error`.
- Concrete exception classes add no additional data members.
- Message construction and storage behavior follows `std::runtime_error` semantics.

---

## Public Interface

## Namespace

```cpp
namespace Exceptions { ... }
```

All exception classes are defined in namespace `Exceptions`.

---

## Base Class

### StrataxError

```cpp
class StrataxError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
```

Base class for all Stratax runtime errors.

---

## Derived Classes

### ShapeError

```cpp
class ShapeError : public StrataxError
{
public:
    using StrataxError::StrataxError;
};
```

Signals an invalid or incompatible shape.

---

### DimensionError

```cpp
class DimensionError : public StrataxError
{
public:
    using StrataxError::StrataxError;
};
```

Signals an invalid dimension count or dimension-arithmetic failure.

---

### IndexError

```cpp
class IndexError : public StrataxError
{
public:
    using StrataxError::StrataxError;
};
```

Signals invalid index access or index normalization failure.

---

### TypeError

```cpp
class TypeError : public StrataxError
{
public:
    using StrataxError::StrataxError;
};
```

Signals unsupported or incompatible type usage.

---

### BroadcastError

```cpp
class BroadcastError : public StrataxError
{
public:
    using StrataxError::StrataxError;
};
```

Signals that broadcasting rules cannot satisfy an operation.

---

### ZeroDivisionError

```cpp
class ZeroDivisionError : public StrataxError
{
public:
    using StrataxError::StrataxError;
};
```

Signals division-by-zero conditions.

---

## Usage Guidelines

- Throw the most specific exception type available for the failure domain.
- Keep messages concrete and operation-local (include expected vs actual values where useful).
- Reserve generic `StrataxError` for cases that do not fit a more specific category.

---

## Complexity Summary

| Operation | Complexity |
| --------- | ----------: |
| Construct exception object with message length `m` | O(m) |
| Throw/catch exception | Implementation-defined |
| Normal path when no exception is thrown | O(1) |

---

## Examples

## Throwing Specific Errors

```cpp
if (index >= size)
{
    throw Exceptions::IndexError("Index is out of bounds.");
}
```

```cpp
if (lhs.shape() != rhs.shape())
{
    throw Exceptions::ShapeError("Shapes must match for elementwise operation.");
}
```

---

## Catching by Base Type

```cpp
try
{
    // Stratax operation
}
catch (const Exceptions::StrataxError& e)
{
    std::cerr << e.what() << '\n';
}
```

---

## Design Notes

The hierarchy is intentionally shallow and message-only to keep exception objects lightweight and predictable.

Using standard runtime-error constructor inheritance (`using ...::runtime_error`) avoids duplicate boilerplate and ensures consistent message behavior across all categories.

---

## Future Improvements

- Consider migration path from `Exceptions` to a nested `stratax::core` namespace if desired.
- Add direct unit tests asserting exception category guarantees per public API.
- Standardize message format conventions across modules.

---

## See Also

- `include/stratax/core/validation/ShapeValidation.hpp`
- `include/stratax/core/validation/DimensionValidation.hpp`
- `include/stratax/core/validation/IndexValidation.hpp`
- `include/stratax/core/validation/TypeValidation.hpp`
