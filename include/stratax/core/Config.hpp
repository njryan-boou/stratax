/** @file
 * @brief Allocation alignment and reserved execution flags.
 */
#pragma once

#include <cstddef>

namespace stratax::core::config {

/** @brief Default alignment used by Stratax allocations. */
inline constexpr std::size_t default_alignment = 64;

/** @brief Reserved bounds-checking flag; current accessors do not consult it. */
inline constexpr bool bounds_checking = true;

/** @brief Reserved SIMD flag; no current operations dispatch on this value. */
inline constexpr bool use_simd = true;

/** @brief Reserved threading flag; no current operations dispatch on this value. */
inline constexpr bool use_multithreading = false;

/** @brief Reserved CUDA flag; no current operations dispatch on this value. */
inline constexpr bool use_cuda = false;

}
