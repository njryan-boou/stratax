@page architecture Architecture

# Architecture

Stratax is a C++20 header-first array library with a compiled pybind11 layer.

## Source layout

| Directory | Responsibility |
| --- | --- |
| include/stratax/core | Buffer ownership, Shape/Slice metadata, ArrayBase, ArrayView, dtype/result traits |
| include/stratax/containers | Fixed-rank Vector/Matrix and arbitrary-rank Tensor |
| include/stratax/indexing | Checked index normalization, unchecked offsets, shared slice views |
| include/stratax/ops | Broadcasting, arithmetic, bitwise, comparison, standard math wrappers |
| include/stratax/algorithms | Creation, copying conversions, reshape/flatten, reductions |
| include/stratax/io | Logical stream formatting |
| bindings | Python argument conversion, lifetime handling, and registration |
| python/stratax | Public exports and typing declarations |
| tests/cpp, tests/python | Behavioral and invariant coverage |

## Ownership and invariants

ArrayBase owns element, shape, and stride buffers together. Copy assignment is
transactional. Vector and Matrix preserve fixed rank across moves, which can
allocate empty metadata. Tensor/base moves leave rank-zero empty sources.
Rank-zero arrays have no scalar storage.

ArrayView copies metadata and borrows an allocation. C++ callers maintain its
lifetime; Python slice wrappers hold the owner. Strides are unsigned element
counts, so reverse views are unsupported. Logical iteration differs from
physical contiguity for views.

## Operations and validation

The structural Array concept admits views, but owning result traits are
specialized for Vector/Matrix/Tensor. Conversions, reshape, flatten, and
reductions accept C++ views; Python binds a smaller surface. Result promotion
is separate from scalar C++ expression evaluation.

Broadcasting aligns trailing dimensions and also rejects empty storage supplying
nonempty results. Compound operations preserve the left shape and dtype.
Checks are local to construction/access/operation boundaries. The validation
compatibility headers contain no functions. Checked at access and unchecked
owning []/() have different contracts; config flags do not override them.

Axis reduction currently copies input to Tensor, then passes strided slices to
callbacks. This preserves input ownership but adds allocation and metadata cost.
Optimization must preserve the documented shape, dtype, lifetime, and exception
contracts.

## Verification and planned work

CI configures C++ jobs on Linux, Windows, and macOS; Python package tests on
Linux for 3.10 through 3.14; and Clang address/undefined-behavior/leak checks.
Local audit results are recorded separately in @ref test_audit.
SIMD/thread/CUDA flags are reserved, and linear algebra, broader statistics,
random sampling, and file I/O remain planned. See @ref roadmap.
