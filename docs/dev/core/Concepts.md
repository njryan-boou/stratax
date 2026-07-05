# Concepts

Developer notes for `include/stratax/core/Concepts.hpp`.

## Purpose

Defines compile-time constraints used by containers and generic algorithms.

## Main API

### Numeric Categories
- `Integral`
- `Floating`
- `Arithmetic`
- `Complex`
- `Numeric`

### Container Categories
- `NDarray`

## Validation Notes

- Concepts fail at compile time, not runtime.
- `Numeric` excludes bool and character-like integer types.
- `NDarray` is structural and checks for the expected array interface.

## Implementation Notes

- Keep concepts small and readable.
- `Arithmetic.hpp` still uses `NDarray` and `Arithmetic` for scalar/container overloads.
- Container element types are constrained through `Numeric`.

## Future Work

- Decide whether concepts should live inside `stratax::core`.
- Add concepts for matrix-like and tensor-like containers.
- Add tests for rejected container-like impostors.
- Add type promotion concepts.
