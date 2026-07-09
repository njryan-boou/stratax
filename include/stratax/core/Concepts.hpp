#pragma once

#include <concepts>
#include <complex>
#include <type_traits>

#include "Types.hpp"

namespace stratax::core::concept_detail {

template<typename T>
using clean_t = std::remove_cvref_t<T>;

}

template<typename T>
concept Integral =
    std::integral<stratax::core::concept_detail::clean_t<T>> &&
    !std::same_as<stratax::core::concept_detail::clean_t<T>, bool> &&
    !std::same_as<stratax::core::concept_detail::clean_t<T>, char> &&
    !std::same_as<stratax::core::concept_detail::clean_t<T>, signed char> &&
    !std::same_as<stratax::core::concept_detail::clean_t<T>, unsigned char> &&
    !std::same_as<stratax::core::concept_detail::clean_t<T>, wchar_t> &&
    !std::same_as<stratax::core::concept_detail::clean_t<T>, char8_t> &&
    !std::same_as<stratax::core::concept_detail::clean_t<T>, char16_t> &&
    !std::same_as<stratax::core::concept_detail::clean_t<T>, char32_t>;

template<typename T>
concept Floating =
    std::floating_point<stratax::core::concept_detail::clean_t<T>>;

template<typename T>
concept Arithmetic =
    Integral<T> ||
    Floating<T>;

template<typename T>
concept Complex =
    std::same_as<stratax::core::concept_detail::clean_t<T>, std::complex<float>> ||
    std::same_as<stratax::core::concept_detail::clean_t<T>, std::complex<double>> ||
    std::same_as<stratax::core::concept_detail::clean_t<T>, std::complex<long double>>;

template<typename T>
concept Numeric =
    Arithmetic<T> ||
    Complex<T>;

namespace stratax::container {

template<typename T>
requires Numeric<T>
class Vector;

template<typename T>
requires Numeric<T>
class Matrix;

template<typename T>
requires Numeric<T>
class Tensor;

}

template<typename T>
struct is_array : std::false_type {};

template<typename T>
requires Numeric<T>
struct is_array<stratax::container::Vector<T>> : std::true_type {};

template<typename T>
requires Numeric<T>
struct is_array<stratax::container::Matrix<T>> : std::true_type {};

template<typename T>
requires Numeric<T>
struct is_array<stratax::container::Tensor<T>> : std::true_type {};

template<typename T>
concept Array =
    is_array<std::remove_cvref_t<T>>::value;

template<typename T>
concept NDarray =
requires(std::remove_reference_t<T>& a)
{
    a.shape();
    a.size();

    a.begin();
    a.end();
};
