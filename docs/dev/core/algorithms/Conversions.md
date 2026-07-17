# Conversions

Developer notes for `include/stratax/core/algorithms/Conversions.hpp`.

## Purpose

Provides helpers for converting between Stratax container shapes and casting container element types.

## Main API

### Shape Conversions
- `to_vector(array)`
- `to_matrix(array)`
- `to_tensor(array)`

### Python Free Functions
- `stratax.to_vector(array)`
- `stratax.to_matrix(array)`
- `stratax.to_tensor(array)`

### Type Casting
- `astype<To>(vector)`
- `astype<To>(matrix)`
- `astype<To>(tensor)`

## Invariants

- Conversion helpers return new owning containers.
- Shape-preserving conversions keep element order in flat storage order.
- `to_vector` only produces rank-1 vectors.
- `to_matrix` preserves rank-2 matrices as-is and collapses singleton dimensions from higher-rank inputs.
- `to_tensor` preserves the input shape exactly.
- `astype` preserves container shape and element count.

## Validation Notes

- `to_vector` requires rank 1 input through `core::validation::require_rank()`.
- `to_matrix` accepts rank-2 input directly or higher-rank input with exactly two non-singleton dimensions.
- `to_tensor` accepts any supported array rank.
- Invalid rank conversions throw `Exceptions::DimensionError`.
- `astype` requires both source and destination element types to satisfy `Numeric`.
- Cast behavior follows `static_cast`.

## Implementation Notes

- Conversion helpers preserve shape and copy linear buffer values with `operator[]`.
- `to_vector` depends on the destination constructor from `Shape`.
- `to_matrix` derives its destination `Shape` by removing singleton dimensions.
- `astype` supports `Vector`, `Matrix`, and `Tensor`.
- Returned containers own their own buffers.
- Rank validation should stay routed through `Validation.hpp`.
- Python bindings expose conversions as module-level free functions, not instance methods.

## Python Usage

```python
from stratax import Matrix, Tensor, Vector, to_matrix, to_tensor, to_vector

vector = Vector([1.0, 2.0, 3.0, 4.0])
matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])
tensor = Tensor([1, 2, 1, 2], 1.0)

reshaped_tensor = to_tensor(vector)
flat_vector = to_vector(tensor)
as_matrix = to_matrix(tensor)
```

## Time Complexity

- `to_vector`, `to_matrix`, and `to_tensor` are `O(n + r)`.
- `astype` is `O(n + r)`.
- Rank checks are `O(1)`.

## Future Work

- Add explicit narrowing or lossy-cast policy if needed.
