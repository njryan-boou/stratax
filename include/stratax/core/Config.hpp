#pragma once

#include <cstddef>

namespace stratax::core::config {

/** @brief Default alignment used by Stratax allocations. */
inline constexpr std::size_t default_alignment = 64;

/** @brief Enables or disables bounds checking in array operations. */
inline constexpr bool bounds_checking = true;

/** @brief Controls whether SIMD code paths are enabled. */
inline constexpr bool use_simd = true;

/** @brief Controls whether multithreaded code paths are enabled. */
inline constexpr bool use_multithreading = false;

/** @brief Controls whether CUDA code paths are enabled. */
inline constexpr bool use_cuda = false;

}
