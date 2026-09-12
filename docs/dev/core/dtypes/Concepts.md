@page concepts DType and Array Concepts

# DType and Array Concepts {#dev_concepts}

Header: `include/stratax/core/dtypes/Concepts.hpp`.
Public concepts are currently declared in the global namespace.

| Concept or trait | Meaning |
| --- | --- |
| `Integral<T>` | One of the supported int8..int64 or uint8..uint64 aliases, excluding bool |
| `Numeric<T>` | Integral, float/double/long double, or supported complex dtype |
| `DType<T>` | Numeric or bool |
| `Ordered<T>` | Supported non-complex dtype, including bool |
| `RealNumeric<T>` | Supported integer or standard floating-point type, excluding bool/complex |
| `Array<T>` | Structural metadata, indexing, and input-range interface with a supported value_type |
| `is_array<T>` | Exact owning Vector, Matrix, or Tensor specialization |

Concepts remove cv/ref qualifiers for classification. is_array does not.
Integer eligibility uses exact fixed-width alias identity; signed/unsigned char
are accepted when they are the platform's int8/uint8 types. There is no
CharacterLike helper and no general exclusion of every character-like type.
Use the supported aliases when portability matters.

## Structural requirements and limits

Array requires size/empty/rank queries, shape/strides access, flat indexed values
convertible to value_type, and begin/end forming an input range. Logical
row-major order is the semantic contract; concepts cannot prove layout,
bounds, ownership, or lifetime. Array does not require mutability or fill().

Views satisfy Array, but that alone does not make every operation available.
Allocating result helpers also require RebindArray or PromoteArray, specialized
for owning containers. Scalar std math calls impose their own admissible-type
requirements. Ordered does not make NaNs totally ordered.

```cpp
#include <stratax.h>

static_assert(Integral<stratax::dtype::int8>);
static_assert(Numeric<const double&>);
static_assert(DType<bool> && !Numeric<bool>);
static_assert(!Ordered<stratax::dtype::complex64>);
static_assert(Array<stratax::core::ArrayView<const double>>);
static_assert(is_array<stratax::container::Vector<double>>::value);
static_assert(!is_array<const stratax::container::Vector<double>>::value);

int main() {}
```

These are compile-time classifications with no runtime validation. Internal
classification helpers live in `stratax::core::concept_detail`.
See @ref types and @ref arrayview.
