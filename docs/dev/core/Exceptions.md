@page exceptions Exceptions

# Exceptions {#dev_exceptions}

Version: v0.2.0

Status: Complete

Umbrella header: `include/stratax/exceptions/Exceptions.hpp`

Component headers:

- `ErrorCode.hpp`: stable error categories
- `StrataxError.hpp`: common message, code, and source-location metadata
- `LayoutErrors.hpp`: shape, dimension, and broadcasting errors
- `IndexErrors.hpp`: index and axis errors
- `TypeErrors.hpp`: type and value-domain errors
- `ArithmeticErrors.hpp`: division-by-zero and overflow errors

---

## Overview

`Exceptions.hpp` aggregates Stratax's domain-specific runtime error hierarchy.
Each family header can also be included independently to reduce dependencies.

All Stratax exceptions derive from `std::runtime_error` through `Exceptions::StrataxError` and are used to communicate shape, dimension, indexing, typing, broadcasting, and arithmetic-domain failures.

---

## Responsibilities

The exception module is responsible for:

- Providing a common base type for Stratax runtime failures
- Defining specific exception categories for common error domains
- Carrying human-readable messages and structured failure metadata
- Exposing a stable error code and category through the common base class
- Capturing the C++ source location where the exception was created

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
    ├── Exceptions::ZeroDivisionError
    ├── Exceptions::AxisError
    ├── Exceptions::OverflowError
    └── Exceptions::ValueError
```

Depends on:

- C++ standard library `<stdexcept>`

Used by:

- Container classes (`Buffer`, `Shape`, `Vector`, `Matrix`, `Tensor`)
- Validation helpers (`validation/*`)
- Operation helpers (`ops/*`, `algorithms/*`)

---

## Invariants

The following conditions are always true:

- Every Stratax-specific exception derives from `Exceptions::StrataxError`.
- `Exceptions::StrataxError` derives from `std::runtime_error`.
- Every exception exposes its domain through `code()` and `category()`, even
  when caught as `const StrataxError&`.
- Direct constructors and named factories capture their caller through
  `std::source_location`; the location is not appended to `what()`.
- Message-only exceptions follow `std::runtime_error` storage semantics.
- Structured constructors retain the values that caused validation to fail.
- Named factories own operation-specific diagnostic text and retain relevant
  metadata, keeping message wording out of validation and operation code.
- Factory-generated messages identify the failed operation, include available
  offending values and expected constraints, explain why the input is invalid,
  and state the valid range or corrective action when one exists.
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
    explicit StrataxError(
        std::string message,
        std::source_location location = std::source_location::current());

    ErrorCode code() const noexcept;
    std::string_view category() const noexcept;
    const std::source_location& location() const noexcept;
    std::string_view file_name() const noexcept;
    std::string_view function_name() const noexcept;
    std::uint_least32_t line() const noexcept;
    std::uint_least32_t column() const noexcept;
    std::string diagnostic() const;
};
```

Base class for all Stratax runtime errors. It owns the diagnostic message and a
stable `ErrorCode`. `category()` returns a lowercase, serialization-friendly
name such as `"shape"`, `"index"`, or `"overflow"`.
`diagnostic()` combines the category, message, and available source location in
a log-ready string while leaving `what()` unchanged.

### ErrorCode

```cpp
enum class ErrorCode
{
    Stratax,
    Shape,
    Dimension,
    Index,
    Type,
    Broadcast,
    ZeroDivision,
    Axis,
    Overflow,
    Value
};
```

Derived constructors assign their corresponding code automatically, including
message-only compatibility constructors and named diagnostic factories.
Every constructor and factory also has a trailing defaulted
`std::source_location` parameter. Factories forward it explicitly so the
recorded location identifies the caller rather than the factory body.

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

Structured instances expose `actual_shape()` and, when applicable,
`expected_shape()` as optional vectors.

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

Structured instances expose optional `actual_dimension()` and
`expected_dimension()` values.

---

### IndexError

```cpp
class IndexError : public StrataxError
{
public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using StrataxError::StrataxError;

    IndexError(difference_type index, size_type size);
    IndexError(std::string message, difference_type index, size_type size);

    bool has_index_metadata() const noexcept;
    const std::optional<difference_type>& index() const noexcept;
    const std::optional<size_type>& size() const noexcept;
};
```

Signals invalid index access or index normalization failure. Bounds errors store
the rejected signed index and indexed extent. Errors describing rank mismatch,
invalid slice configuration, or another non-bound condition retain support for
message-only construction and report empty metadata optionals.

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

Structured instances expose optional `actual_type()` and `expected_type()`
names.

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

Structured instances expose optional `left_shape()` and `right_shape()` values.

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

When a failing divisor belongs to an array, structured instances expose its
flat `index()`. Scalar and aggregate failures may remain message-only.

---

### AxisError

Signals an invalid reduction axis. Structured instances expose the signed
`axis()` and array `rank()`.

---

### OverflowError

Signals checked arithmetic overflow. Structured instances expose the
`operation()`, `left_operand()`, and `right_operand()`.

---

### ValueError

Signals a value that belongs to the correct type but falls outside an
operation's accepted domain, such as an invalid bit-shift count.

---

## Usage Guidelines

- Throw the most specific exception type available for the failure domain.
- Prefer a named exception factory for known failure modes.
- Add new diagnostic text to the relevant exception class instead of a throw
  site, then pass structured values from the caller.
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
throw Exceptions::IndexError(index, size);
```

```cpp
throw Exceptions::BroadcastError::incompatible(
    {lhs.shape().begin(), lhs.shape().end()},
    {rhs.shape().begin(), rhs.shape().end()});
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
    std::cerr << e.category() << ": " << e.what() << '\n';
    std::cerr << e.file_name() << ':' << e.line() << '\n';
}
```

---

## Design Notes

The hierarchy remains shallow. Every concrete exception retains inherited
message-only construction for failures without meaningful structured values.
Metadata-bearing constructors generate consistent default messages. Named
factories cover operation-specific wording, so library and binding call sites
do not duplicate diagnostic strings.

---

## Future Improvements

- Consider migration path from `Exceptions` to a nested `stratax::core` namespace if desired.
- Standardize message format conventions across modules.

---

## See Also

- `include/stratax/core/validation/IndexValidation.hpp`
- `include/stratax/core/validation/Validation.hpp`
