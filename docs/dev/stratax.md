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
- Simplifying consumer include management

The umbrella header is not responsible for:

- Declaring new APIs itself
- Runtime initialization behavior
- Build or link configuration

---

## Aggregated Components

### Core Foundations

- `include/stratax/core/containers/Buffer.hpp`
- `include/stratax/core/Concepts.hpp`
- `include/stratax/core/Config.hpp`
- `include/stratax/core/Exceptions.hpp`
- `include/stratax/core/containers/Shape.hpp`
- `include/stratax/core/containers/Strides.hpp`
- `include/stratax/core/Types.hpp`
- `include/stratax/core/validation/Validation.hpp`
- `include/stratax/core/Slice.hpp`

### Containers

- `include/stratax/core/containers/Matrix.hpp`
- `include/stratax/core/containers/Tensor.hpp`
- `include/stratax/core/containers/Vector.hpp`

### Algorithms

- `include/stratax/core/algorithms/Creation.hpp`
- `include/stratax/core/algorithms/Conversions.hpp`
- `include/stratax/core/algorithms/Reshape.hpp`
- `include/stratax/core/algorithms/Reductions.hpp`

### Ops

- `include/stratax/core/ops/Comparison.hpp`
- `include/stratax/core/ops/Arithmetic.hpp`
- `include/stratax/core/ops/Indexing.hpp`
- `include/stratax/core/ops/Slice.hpp`

### I/O

- `include/stratax/core/io/Print.hpp`

---

## Usage

```cpp
#include <stratax.h>

int main()
{
    stratax::container::Vector<double> v{1.0, 2.0, 3.0};
    auto t = to_tensor(v);
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
