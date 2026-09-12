@page test_audit Test and Invariant Audit

# Test and invariant audit — 2026-09-11

The baseline had 488 passing C++ tests and 20 failing Python tests out of 151.
Those numbers overstated confidence: the Python path setup selected `tests/`
as the repository root and could load a stale installed extension; Python CI
only checked import; five runtime tests asserted `true` for empty compatibility
headers; and the public-header test file was empty.

The audit reviewed test discovery and CI, storage ownership, shape/stride
consistency, checked indexing, views, conversions, arithmetic/broadcasting,
reductions, and the Python bindings. It added boundary and reference-based
checks rather than treating a passing example as proof of an invariant.

## Defects fixed

| Area | Failure | Regression coverage |
| --- | --- | --- |
| Buffer construction | Throwing default construction leaked the raw allocation. | Element lifetime counters and LeakSanitizer; partial copy/fill/list failures. |
| Array copy assignment | Allocation failure during metadata copying could leave storage and shape inconsistent. | Transactional copy/swap; existing copy tests plus a separate allocation-failure probe at all three allocation points. |
| Vector/Matrix moves | Moved-from objects lost their fixed rank; matrix dimension access could dereference absent metadata. | Move construction, assignment, source reuse, destination ownership, and complete layout checks. |
| Shape element count | An overflowing prefix threw before a later zero dimension was reached. | Every permutation of `{0, 2, SIZE_MAX}`. |
| Slice arithmetic | Extreme steps overflowed signed rounding/negation arithmetic and produced incorrect counts. | Full signed-domain raw ranges, normalized extreme steps, and 1,792 Python slice/list comparisons. |
| Empty slices | Empty matrix/tensor slices could form pointers beyond the allocation; empty vectors performed null-pointer arithmetic. | Empty views retain the source base pointer; null and multidimensional empty cases. |
| View construction | Mismatched ranks, null nonempty storage, overflowing element counts and offsets were accepted. | Invalid metadata rejected at construction, before noexcept size queries or indexing. |
| Broadcasting | Rank-zero empty operands could be indexed as if they contained a scalar. | Arithmetic, comparisons, bitwise/shift operations, both operand orders, and compound assignments. |
| Conversion | Removing a zero axis could create a nonempty matrix from an empty tensor. | Only singleton axes are removed; zero extents and element counts are preserved. |
| Compound division | A later zero divisor left earlier values modified. The old test masked this by dividing earlier values by one. | Divisor changed to `{2, 0}`; C++ and Python now check unchanged values and shape on failure. |
| Python matrix slicing | Top-level row slices were parsed as integer indices. | Existing row-slice test restored; zero-step rejection also checked. |
| Python axis indices | `argmin`/`argmax` returned an unregistered integer tensor type. | `IndexTensor` binding, every axis/keepdims combination, exact integer values, indexing and iterator lifetime. |

## Coverage added or strengthened

- Buffer deep copies, moves without element copies, source reuse, self-assignment,
  swap, alignment, allocation-size overflow, reverse iteration, cleanup, and the
  strong copy-assignment guarantee when element copies throw.
- Container storage/shape/stride consistency before and after ownership changes.
- Noncontiguous view iteration and checked negative indexing; shared writes;
  Python view owner lifetime, metadata, empty slices, and invalid argument types.
- Broadcast results checked against independent coordinate calculations,
  including noncommutative operations, mixed container families, zero extents,
  incompatible shapes, and in-place identity/shape preservation.
- All nine reductions checked against independent Python calculations across
  three axes, negative aliases of those axes, and both keepdims settings.
  Empty identities, undefined empty reductions, empty output domains, invalid
  axes, first-index tie behavior, and input immutability are covered.
- Source and installed-package tests verify the actual package and extension
  paths. Missing source extensions no longer silently fall back to an installed
  binary. CI runs pytest for its Python version matrix and enables ASan, UBSan,
  and leak checking in its Clang job.
- The empty public-header test now exercises an end-to-end example. Config and
  iterator placeholders now assert behavior. The five empty validation headers
  retain compile-only checks instead of inflating the runtime test count.

Existing dtype traits/promotion, creation, reshape, printing, comparison,
bitwise and math tests remain active. No tests are disabled or marked xfail.

## Validation

- GCC Debug/CTest: **506 passed**.
- Clang Debug with address, undefined-behavior and leak sanitizers: **506 passed**.
- Python 3.13 source extension: **251 passed**.
- Freshly built Python 3.13 wheel, installed separately from the editable
  environment: **251 passed**.
- All **36 public headers** compiled individually with C++20.
- A standalone allocation-failure probe confirmed that each of the three
  aligned allocations during matrix copy assignment can fail without changing
  destination storage, values, shape, or strides. This probe was an audit check,
  not a newly registered CI test.

Normal reruns:

```sh
cmake --build build
ctest --test-dir build --output-on-failure
python -m pytest tests/python
python -m pytest tests/python --installed
```

Source Python tests require a current extension in `python/stratax/`; configure
CMake with `STRATAX_BUILD_PYTHON_BINDINGS=ON` and the intended Python interpreter,
then build `_core`. Installed mode requires installing the current wheel first.

## Limits and follow-up work

This is stronger regression coverage, not proof of completeness or a measured
100% branch-coverage result. Windows/macOS and Python versions other than 3.13
were not executed locally; the configured CI matrix still needs to run.

- Negative-step views remain explicitly unsupported by the current C++ design.
  Python expectations now match that existing contract. Implementing reverse
  views requires signed strides; slicing an existing Python view also remains
  unsupported. README and the Python guide now state these limits.
- Exhaustive overlapping-view compound-operation semantics, NaN/Inf behavior,
  signed arithmetic overflow, and every dtype-pair combination are not certified
  by this audit. Those need focused contracts and additional adversarial tests.
- Raw C++ views cannot verify the length or lifetime of the allocation supplied
  by a caller. Constructor validation checks metadata, not external ownership.
- Preserving Vector/Matrix rank after a move requires allocating empty shape
  metadata. These moves may now throw `std::bad_alloc`; they are no longer
  noexcept. Buffer and ArrayBase moves remain noexcept.

Subsequent documentation audits updated the detailed developer references to
match shared views, current namespaces, and the contracts above.
