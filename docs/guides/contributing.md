@page contributing Contributing

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
- Explain actual behavior, invariants, failures, ownership, and relevant costs;
  let generated header docs supply exact declarations.
- Update user-facing guides under `docs/guides` when behavior changes.
- Regenerate Doxygen output intentionally; do not hand-edit files under `docs/output/html`.
- Keep API status notes aligned with implementation.
- Keep troubleshooting and release notes current when build, packaging, or docs deployment changes.
- Keep the Python API reference aligned with `python/stratax/__init__.py` and the `.pyi` stubs.

### Verify documentation examples

After rebuilding the source extension, run:

```sh
python scripts/check-doc-examples.py
```

Use complete, independent C++/Python examples with assertions for important
results. Label signature sketches and pseudocode as text instead of executable
code. Check negative indices, empty dimensions, mutation, and result types when
they affect the example. Keep planned features explicitly marked as planned.
Doxygen generation and local link checks complement execution; none alone proves
that prose is accurate.

The CI matrix also compiles every public header independently and publishes
C++ line/branch coverage artifacts. Doxygen warnings fail the documentation
build. See @ref verification for local commands and how to interpret coverage,
and @ref numerical_contract for arithmetic preconditions and precision limits.
Performance work should use @ref benchmarks to compare equivalent workloads.

### Naming

- Classes use `PascalCase`.
- Functions use `snake_case`.
- Concepts use `PascalCase`.
- Follow nearby naming conventions; config constants use snake_case.

## Validation Notes

- Changes that affect shape, indexing, construction, or arithmetic should include edge-case tests.
- Avoid adding validation behavior in docs unless the implementation already supports it or it is clearly marked as future work.
- Keep validation checks local to the operation whose preconditions they enforce.

## Implementation Notes

- Prefer small, local changes over broad refactors.
- Reuse existing container interfaces instead of introducing parallel APIs.
- Keep generated or build artifacts out of commits.
- Do not change unrelated files while addressing a focused task.

## Future Work

- Add formatting and linting rules.
- Add pull request checklist templates.
