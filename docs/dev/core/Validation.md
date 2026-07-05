# Validation

Developer notes for `include/stratax/core/Validation.hpp`.

## Purpose

Reserved for shared runtime validation helpers, but intentionally empty for now.

## Main API

### Current State
- No active validation helpers.
- Containers and ops should not depend on this file yet.

### Planned Helpers
- Shape validation.
- Rank validation.
- Bounds validation.
- Broadcasting validation.
- Matrix dimension validation.

## Validation Notes

- Validation work is deferred by design.
- Tests should not expect validation behavior until this file is reintroduced.
- Code may still use direct exception checks where already implemented.

## Implementation Notes

- Keep this file empty until validation design is ready.
- When restored, helpers should be small and header-only.
- Be careful with `Shape::size()` versus `Shape::rank()`.

## Future Work

- Decide exception types.
- Restore validation tests.
- Add helper coverage for vector, matrix, tensor, and ops.
- Add broadcasting validation.
