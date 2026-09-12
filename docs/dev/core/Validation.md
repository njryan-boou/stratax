@page validation Validation

# Validation {#dev_validation}

Status: Compatibility headers only

## Overview

Stratax no longer has a validation-helper layer. Preconditions and invariant
checks are expressed as ordinary `if` statements at the point where the value
is used, followed by a simple exception from `Exceptions.hpp`.

This keeps the condition, exception category, and relevant operation together
and avoids a second API that only wraps an `if` statement.

The headers under `include/stratax/core/validation/` remain empty so existing
includes do not immediately break. They define no functions and should not be
used by new code.

Checked signed index normalization remains a real indexing operation rather
than a generic validator and is provided by:

```cpp
#include <stratax.h>
#include <cassert>

int main() {
    assert(stratax::indexing::normalize_index(-1, 4) == 3);
}
```

It accepts positive and Python-style negative indices and throws
`Exceptions::IndexError` when the index is outside the valid range.

Unchecked accessors still have caller preconditions; not every invariant is
checked on every access. The reserved bounds_checking flag does not change this.
