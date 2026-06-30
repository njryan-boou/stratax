#pragma once

#include <cstddef>

namespace core::config {

inline constexpr std::size_t default_alignment = 64;

inline constexpr bool bounds_checking = true;

inline constexpr bool use_simd = true;

inline constexpr bool use_multithreading = false;

inline constexpr bool use_cuda = false;

}
