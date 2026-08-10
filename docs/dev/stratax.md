@page dev_stratax stratax.h

# stratax.h

Version: v0.2.0

Status: Complete

Header: include/stratax.h

---

## Overview

`stratax.h` is the umbrella include for the C++ API surface.

Including this header pulls in core metadata/types, containers, algorithms, operations, and I/O helpers so downstream code can use Stratax without manually including each component header.

---

## Responsibilities

The umbrella header is responsible for:

- Providing a single include entry point for Stratax C++ usage
- Aggregating stable public core headers
- Re-exporting common public types and algorithms under `stratax`
- Simplifying consumer include management

The umbrella header is not responsible for:

- Implementing component APIs directly
- Runtime initialization behavior
- Build or link configuration

---

## Aggregated Components

### Core Foundations

- `include/stratax/core/Buffer.hpp`
- `include/stratax/concepts/Numeric.hpp`
- `include/stratax/core/Config.hpp`
- `include/stratax/exceptions/Exceptions.hpp`
- `include/stratax/core/Shape.hpp`
- `include/stratax/core/Strides.hpp`
- `include/stratax/core/Types.hpp`
- `include/stratax/core/validation/Validation.hpp`
- `include/stratax/core/Slice.hpp`

### Containers

- `include/stratax/containers/Matrix.hpp`
- `include/stratax/containers/Tensor.hpp`
- `include/stratax/containers/Vector.hpp`

### Algorithms

- `include/stratax/algorithms/Creation.hpp`
- `include/stratax/algorithms/Conversion.hpp`
- `include/stratax/algorithms/Reshape.hpp`
- `include/stratax/algorithms/Reductions.hpp`

### Ops

- `include/stratax/ops/Comparison.hpp`
- `include/stratax/ops/Arithmetic.hpp`
- `include/stratax/indexing/Broadcasting.hpp`
- `include/stratax/indexing/Indexing.hpp`
- `include/stratax/indexing/Slicing.hpp`

### I/O

- `include/stratax/io/Print.hpp`

---

## Usage

```cpp
#include <stratax.h>

int main()
{
    stratax::Vector<double> v{1.0, 2.0, 3.0};
    auto t = stratax::to_tensor(v);
    (void)t;
}
```

---

## Design Notes

This header intentionally acts as a convenience aggregator. Component-level headers should still be preferred in translation units that need tighter include boundaries or reduced compile-time impact.

---

## Future Improvements

- Add policy guidance on when to prefer umbrella vs component includes
- Keep include ordering aligned with logical module groupings as APIs expand

---

## See Also

- `docs/dev/README.md`
- `docs/dev/core/Concepts.md`
- `docs/dev/core/containers/Vector.md`
