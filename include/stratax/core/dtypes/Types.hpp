/** @file
 * @brief Supported scalar dtype aliases.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <complex>

namespace stratax::core
{

    /** @brief Unsigned size and index type. */
    using index_t = std::size_t;
    /** @brief Signed pointer-difference and index type. */
    using ssize_t = std::ptrdiff_t;

}

namespace stratax::dtype
{

    /** @brief Boolean dtype. */
    using bool_ = bool;

    /** @brief Signed 8-bit integer dtype. */
    using int8  = std::int8_t;
    /** @brief Signed 16-bit integer dtype. */
    using int16 = std::int16_t;
    /** @brief Signed 32-bit integer dtype. */
    using int32 = std::int32_t;
    /** @brief Signed 64-bit integer dtype. */
    using int64 = std::int64_t;

    /** @brief Unsigned 8-bit integer dtype. */
    using uint8  = std::uint8_t;
    /** @brief Unsigned 16-bit integer dtype. */
    using uint16 = std::uint16_t;
    /** @brief Unsigned 32-bit integer dtype. */
    using uint32 = std::uint32_t;
    /** @brief Unsigned 64-bit integer dtype. */
    using uint64 = std::uint64_t;

    /** @brief Native float dtype. */
    using float32 = float;
    /** @brief Native double dtype. */
    using float64 = double;
    /** @brief Native long double dtype; precision is platform-dependent. */
    using longdouble = long double;

    /** @brief Complex dtype with native float components. */
    using complex64  = std::complex<float>;
    /** @brief Complex dtype with native double components. */
    using complex128 = std::complex<double>;
    /** @brief Complex dtype with native long double components. */
    using clongdouble = std::complex<long double>;
}
