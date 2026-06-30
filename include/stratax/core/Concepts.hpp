#pragma once

#include <concepts>
#include <complex>
#include <type_traits>

#include "Types.hpp"
namespace stratax::core
{
    template<typename T>
    concept Integral = std::integral<T>;

    template<typename T>
    concept Floating = std::floating_point<T>;

    template<typename T>
    concept Complex =
        std::same_as<T, std::complex<float>> ||
        std::same_as<T, std::complex<double>>;

    template<typename T>
    concept Numeric =
        Integral<T> ||
        Floating<T> ||
        Complex<T>;
}
