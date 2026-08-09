#pragma once

#include <concepts>
#include <complex>
#include <type_traits>

#include "Types.hpp"

namespace stratax::core::concept_detail {

/** @brief Removes cv-qualifiers and references from a type. */
template<typename T>
using clean_t = std::remove_cvref_t<T>;

template<typename T, typename... Candidates>
concept SameAsAny =
    (std::same_as<clean_t<T>, Candidates> || ...);

template<typename T>
concept CharacterLike =
    SameAsAny<
        T,
        char,
        signed char,
        unsigned char,
        wchar_t,
        char8_t,
        char16_t,
        char32_t>;

template<typename T>
concept BoolLike =
    std::same_as<clean_t<T>, bool>;

template<typename T>
concept SupportedComplex =
    SameAsAny<
        T,
        std::complex<float>,
        std::complex<double>,
        std::complex<long double>>;

}

/** @brief Matches signed and unsigned integral types, excluding character-like types. */
template<typename T>
concept Integral =
    std::integral<stratax::core::concept_detail::clean_t<T>> &&
    !stratax::core::concept_detail::BoolLike<T> &&
    !stratax::core::concept_detail::CharacterLike<T>;

/** @brief Alias for Stratax-supported integer types. */
template<typename T>
concept Integer =
    Integral<T>;

/** @brief Matches supported signed integer types. */
template<typename T>
concept SignedInteger =
    Integral<T> &&
    std::signed_integral<stratax::core::concept_detail::clean_t<T>>;

/** @brief Matches supported unsigned integer types. */
template<typename T>
concept UnsignedInteger =
    Integral<T> &&
    std::unsigned_integral<stratax::core::concept_detail::clean_t<T>>;

/** @brief Matches floating-point types. */
template<typename T>
concept Floating =
    std::floating_point<stratax::core::concept_detail::clean_t<T>>;

/** @brief Alias for Stratax-supported floating-point types. */
template<typename T>
concept Float =
    Floating<T>;

/** @brief Matches arithmetic scalar types. */
template<typename T>
concept Arithmetic =
    Integral<T> ||
    Floating<T>;

/** @brief Matches real numeric scalar types. */
template<typename T>
concept Real =
    Arithmetic<T>;

/** @brief Matches supported complex scalar types. */
template<typename T>
concept Complex =
    stratax::core::concept_detail::SupportedComplex<T>;

/** @brief Matches all scalar types supported by Stratax numeric containers. */
template<typename T>
concept Numeric =
    Arithmetic<T> ||
    Complex<T>;

/** @brief Alias for any scalar type accepted by Stratax numeric containers. */
template<typename T>
concept Scalar =
    Numeric<T>;

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

/** @brief Type trait that reports whether a type is a Stratax array. */
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
    is_array<stratax::core::concept_detail::clean_t<T>>::value;

/** @brief Matches array-like container types with shape, size, and iteration support. */
template<typename T>
concept NDarray =
requires(stratax::core::concept_detail::clean_t<T>& a)
{
    a.shape();
    a.size();

    a.begin();
    a.end();
};
