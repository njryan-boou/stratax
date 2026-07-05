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
- `Exceptions::DTypeError`
- `Exceptions::ZeroDivisionError`

## Validation Notes

- Validation helpers are currently deferred.
- Existing containers may still throw these exceptions directly.
- Standard exceptions are still used in some lower-level or generic code.

## Implementation Notes

- Exceptions currently live in namespace `Exceptions`.
- Keep exception classes lightweight.
- Prefer specific exception types when the error category is clear.

## Future Work

- Decide whether exceptions should move under `stratax::core`.
- Normalize standard exceptions versus Stratax exceptions.
- Add tests for exception type guarantees.
- Add messages consistently across containers and ops.
