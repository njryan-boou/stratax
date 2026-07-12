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
- `Array`
- `NDarray`

## Invariants

- `Numeric` excludes `bool` and character-like integer types.
- `Numeric` accepts integral, floating point, and supported `std::complex` types.
- Numeric concepts normalize const/reference qualifiers before classification.
- `Array` only recognizes Stratax `Vector`, `Matrix`, and `Tensor` specializations.
- Concepts should not impose runtime behavior.

## Validation Notes

- Concepts fail at compile time, not runtime.
- `Numeric` excludes bool and character-like integer types.
- Numeric concepts normalize const/reference qualifiers.
- `Array` recognizes Stratax `Vector`, `Matrix`, and `Tensor`.
- `NDarray` is structural and checks for the expected array interface.

## Implementation Notes

- Keep concepts small and readable.
- Arithmetic scalar overloads use `Numeric` so complex scalars are supported.
- Container element types are constrained through `Numeric`.

## Time Complexity

- Concepts are compile-time checks and have no runtime cost.
- `Array` and numeric category checks are constant-size template checks at compile time.
- `NDarray` checks the presence of required member functions at compile time.

## Future Work

- Decide whether concepts should live inside `stratax::core`.
- Add concepts for matrix-like and tensor-like containers.
- Add tests for rejected container-like impostors.
- Add type promotion concepts.
