@page types Dtypes and Promotion

# Dtypes and Promotion {#dev_types}

Headers: `include/stratax/core/dtypes/Types.hpp`, `DTypeTraits.hpp`, and
`Promotion.hpp`; owning container result rules are in `core/ArrayTraits.hpp`.

## Scalar aliases

| Namespace | Names | C++ types |
| --- | --- | --- |
| `stratax::core` | `index_t`, `ssize_t` | size_t, ptrdiff_t |
| `stratax::dtype` | `bool_` | bool |
| `stratax::dtype` | `int8`, `int16`, `int32`, `int64` | Corresponding fixed-width signed integers |
| `stratax::dtype` | `uint8`, `uint16`, `uint32`, `uint64` | Corresponding fixed-width unsigned integers |
| `stratax::dtype` | `float32`, `float64`, `longdouble` | float, double, long double |
| `stratax::dtype` | `complex64`, `complex128`, `clongdouble` | std::complex of those real types |

There are no float128 or complex256 aliases. Floating-point names map to native
C++ representations; long double precision is platform-dependent.

## Metadata and result traits

`stratax::core::DTypeTraits<T>` removes cv/ref qualifiers and exposes type,
kind, storage bits, precision digits, and name. Floating/complex types also have
a promotion rank; complex metadata includes component_type. Storage bits may
include padding. ComplexComponent and ComplexFromReal map between registered
complex and real component dtypes.

`promote_t<L, R>` selects the result dtype through Promote specializations.
Refer to the generated Promotion.hpp reference for category-specific rules;
this is distinct from C++'s intermediate expression type. Allocating operations
usually compute using original scalar types then convert the result.
`rebind_array_t<A, T>` changes the dtype of an owning family;
`promote_array_t<L, R, T>` retains matching families and chooses Tensor for mixed
families. ArrayView has no supplied owning result specialization.

```cpp
#include <stratax.h>
#include <type_traits>

using stratax::dtype::float64;
using stratax::core::index_t;
static_assert(std::is_same_v<float64, double>);
static_assert(std::is_same_v<stratax::core::promote_t<stratax::dtype::int16,
                                                  stratax::dtype::int32>,
                             stratax::dtype::int32>);
int main() {
    float64 value = 3.5;
    index_t count = 4;
    (void)value;
    (void)count;
}
```

All aliases and traits are compile-time constructs. They do not add checked
casts, overflow detection, or runtime dtype dispatch. See @ref concepts.
