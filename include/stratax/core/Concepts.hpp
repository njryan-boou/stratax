#pragma once

#include <concepts>
#include <complex>
#include <type_traits>

#include "Types.hpp"

namespace stratax::core
{

template<typename T>
concept Integral =
    std::integral<T> &&
    !std::same_as<T, bool> &&
    !std::same_as<T, char> &&
    !std::same_as<T, signed char> &&
    !std::same_as<T, unsigned char> &&
    !std::same_as<T, wchar_t> &&
    !std::same_as<T, char8_t> &&
    !std::same_as<T, char16_t> &&
    !std::same_as<T, char32_t>;

template<typename T>
concept Floating =
    std::floating_point<T>;

template<typename T>
concept Complex =
    std::same_as<T, std::complex<float>> ||
    std::same_as<T, std::complex<double>> ||
    std::same_as<T, std::complex<long double>>;

template<typename T>
concept Numeric =
    Integral<T> ||
    Floating<T> ||
    Complex<T>;

} // namespace stratax::core