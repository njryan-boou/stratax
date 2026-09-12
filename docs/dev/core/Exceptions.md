@page exceptions Exceptions

# Exceptions {#dev_exceptions}

Header: `include/stratax/exceptions/Exceptions.hpp`

## Overview

Stratax uses a small exception hierarchy for runtime errors. Each exception
contains only the message supplied when it is thrown. There are no error codes,
metadata objects, source-location fields, or named exception factories.

Checks are performed directly where their invariants matter:

```cpp
#include <stratax.h>

int main() {
const stratax::core::Shape shape{2, 3};
if (shape.rank() != 2)
{
    throw Exceptions::ShapeError("Matrix requires a rank-2 shape.");
}
}
```

## Hierarchy

```text
std::runtime_error
└── Exceptions::StrataxError
    ├── Exceptions::ShapeError
    ├── Exceptions::DimensionError
    ├── Exceptions::RankError
    ├── Exceptions::IndexError
    ├── Exceptions::TypeError
    ├── Exceptions::BroadcastError
    ├── Exceptions::ZeroDivisionError
    ├── Exceptions::AxisError
    ├── Exceptions::OverflowError
    └── Exceptions::ValueError
```

Every derived class inherits the constructors of `StrataxError`, so it can be
constructed directly from a message. Specialized errors can also be caught as
`Exceptions::StrataxError` or `std::runtime_error`.

## Error selection

| Exception | Use |
|---|---|
| `ShapeError` | Invalid shape or incompatible element count |
| `DimensionError` | Invalid dimension or dimension arithmetic overflow |
| `RankError` | Incorrect number of dimensions, indices, or slices |
| `IndexError` | Invalid index or unsupported slice |
| `TypeError` | Unsupported runtime type |
| `BroadcastError` | Incompatible broadcast shapes |
| `ZeroDivisionError` | Division by zero or undefined empty statistic |
| `AxisError` | Invalid reduction axis |
| `OverflowError` | Explicitly reported runtime conversion overflow; not automatic arithmetic checking |
| `ValueError` | Invalid value that does not fit another category |

All exception classes are declared by `Exceptions.hpp`; there are no separate
family headers.

Python registers these classes beneath stratax.StrataxError, but some argument
conversion and slice parsing errors use built-in Python exception classes.
See @ref python_api for that boundary.
