# Architecture

Developer notes for the overall Stratax project layout and design direction.

## Purpose

Describes how the library is organized so implementation work stays consistent across core types, containers, operations, tests, and future bindings.

## Main API

### Project Layout
- `include/stratax/core`
- `include/stratax/containers`
- `include/stratax/algorithms`
- `include/stratax/ops`
- `include/stratax/io`
- `tests`
- `docs`

### Core Model
- Containers own contiguous storage.
- `Shape` describes logical dimensions.
- `Strides` maps multidimensional indexes to flat offsets.
- Algorithms are implemented as free functions where possible.

### Extension Areas
- Linear algebra
- Statistics
- Random number support
- Python bindings
- Serialization and file I/O

## Validation Notes

- Container invariants should be checked at construction boundaries.
- Shape-sensitive algorithms should reject mismatched shapes before touching data.
- Validation helpers are currently deferred until `Validation.hpp` is restored.
- Tests should cover behavior at the public API boundary, not private implementation details.

## Implementation Notes

- The library is header-only C++20.
- Generic code should follow existing concepts and container APIs.
- Containers should expose a consistent indexing style with `operator()`.
- Shared behavior should live in core utilities instead of being duplicated across containers.
- Python binding plans should not force C++ API compromises before the C++ layer is stable.

## Future Work

- SIMD optimization
- Parallel execution
- Python bindings through pybind11
- More complete validation helpers
- GPU support investigation
