#pragma once

#include <cstddef>
#include <cstdint>
#include <complex>

namespace stratax::core::dtype
{
    // Boolean
    
    using bool_ = bool;

    // Signed integers

    using int8  = std::int8_t;
    using int16 = std::int16_t;
    using int32 = std::int32_t;
    using int64 = std::int64_t;

    // Unsigned integers

    using uint8  = std::uint8_t;
    using uint16 = std::uint16_t;
    using uint32 = std::uint32_t;
    using uint64 = std::uint64_t;

    // Floating point

    using float32 = float;
    using float64 = double;
    using float128 = long double;

    // Complex

    using complex64  = std::complex<float>;
    using complex128 = std::complex<double>;
    using complex256 = std::complex<long double>;

    // Indexing

    using index_t = std::size_t;
    using ssize_t = std::ptrdiff_t;
}