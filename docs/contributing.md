# Contributing

Developer notes for making changes to Stratax consistently.

## Purpose

Defines the working conventions for code, tests, and documentation so new changes fit the existing project style.

## Main API

### Code Style

- Use C++20.
- Keep headers self-contained.
- Prefer existing concepts and helper types.
- Use `constexpr` when it improves correctness or clarity.

### Testing

- Add or update tests beside the feature being changed.
- Cover normal behavior and edge cases.
- Keep tests focused on public behavior.

### Documentation

- Update the matching file under `docs/dev` when public behavior or design intent changes.
- Keep developer docs practical and implementation-focused.
- Use the Arithmetic document structure for module docs.

### Naming

- Classes use `PascalCase`.
- Functions use `snake_case`.
- Concepts use `PascalCase`.
- Constants use `UPPER_CASE`.

## Validation Notes

- Changes that affect shape, indexing, construction, or arithmetic should include edge-case tests.
- Avoid adding validation behavior in docs unless the implementation already supports it or it is clearly marked as future work.
- Keep validation expectations aligned with the current state of `Validation.hpp`.

## Implementation Notes

- Prefer small, local changes over broad refactors.
- Reuse existing container interfaces instead of introducing parallel APIs.
- Keep generated or build artifacts out of commits.
- Do not change unrelated files while addressing a focused task.

## Future Work

- Add a full public contribution guide.
- Document build and test commands once the build system is finalized.
- Add formatting and linting rules.
- Add pull request checklist templates.
