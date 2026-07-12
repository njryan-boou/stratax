# Exceptions

Developer notes for `include/stratax/core/Exceptions.hpp`.

## Purpose

Defines the Stratax exception hierarchy used for domain-specific runtime errors.

## Main API

### Base Error
- `Exceptions::StrataxError`

### Shape and Dimension Errors
- `Exceptions::ShapeError`
- `Exceptions::DimensionError`
- `Exceptions::BroadcastError`

### Access and Type Errors
- `Exceptions::IndexError`
- `Exceptions::TypeError`
- `Exceptions::ZeroDivisionError`

## Invariants

- All Stratax-specific exceptions derive from `Exceptions::StrataxError`.
- `Exceptions::StrataxError` derives from `std::runtime_error`.
- Exception classes do not add state beyond the runtime error message.

## Validation Notes

- Validation helpers are currently deferred.
- Existing containers may still throw these exceptions directly.
- Standard exceptions are still used in some lower-level or generic code.

## Implementation Notes

- Exceptions currently live in namespace `Exceptions`.
- Keep exception classes lightweight.
- Prefer specific exception types when the error category is clear.

## Time Complexity

- Constructing an exception object is `O(m)`, where `m` is the message length.
- Throwing/catching cost is implementation-defined and should stay off hot paths.
- Exception type checks have no normal-path runtime cost until an exception is thrown.

## Future Work

- Decide whether exceptions should move under `stratax::core`.
- Normalize standard exceptions versus Stratax exceptions.
- Add tests for exception type guarantees.
- Add messages consistently across containers and ops.
