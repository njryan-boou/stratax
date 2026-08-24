@page validation Validation

# Validation {#dev_validation}

Status: Active

Headers:

- `include/stratax/core/validation/Validation.hpp`
- `include/stratax/core/validation/DimensionValidation.hpp`
- `include/stratax/core/validation/IndexValidation.hpp`
- `include/stratax/core/validation/ShapeValidation.hpp`
- `include/stratax/core/validation/TypeValidation.hpp`

---

## Overview

The validation module centralizes reusable dimension, index, shape, and runtime
type checks. Failures use metadata-bearing exceptions whose constructors own
the diagnostic wording. `Validation.hpp` aggregates all four validation
headers.

## Public Interface

- `valid_index(index, size)` reports whether `index` is in `[0, size)` or the
  equivalent negative range `[-size, -1]`.
- `require_valid_index(index, size)` throws `Exceptions::IndexError` when the
  index is invalid.
- `normalize_index(index, size)` validates the index and returns its nonnegative
  zero-based position.
- `require_nonnegative_dimension(value)` rejects negative dimensions.
- `require_dimension(actual, expected)` checks dimension equality.
- `require_same_shape(actual, expected)` checks shape equality.
- `require_type(matches, actual, expected)` reports runtime type
  mismatches while retaining both type names.

## Invariants

- Every validation header can be included independently.
- Negative-index normalization cannot overflow, including for `PTRDIFF_MIN`.
- Invalid indices preserve the attempted index and valid extent in the thrown
  `Exceptions::IndexError` metadata.
- A zero-sized extent rejects every index.
- Dimension, shape, and type failures preserve their actual and expected values.
- Validation call sites do not construct or duplicate diagnostic messages.
