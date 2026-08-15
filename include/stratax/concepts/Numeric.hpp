#pragma once

#include <concepts>
#include <complex>
#include <type_traits>

#include <stratax/core/Types.hpp>

namespace stratax::core::concept_detail {



template<typename T, typename... Candidates>
concept SameAsAny =
	(std::same_as<std::remove_cvref_t<T>, Candidates> || ...);

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
	std::same_as<std::remove_cvref_t<T>, bool>;

template<typename T>
concept SupportedComplex =
	SameAsAny<
		T,
		std::complex<float>,
		std::complex<double>,
		std::complex<long double>>;

}

template<typename T>
concept Integral =
    std::integral<std::remove_cvref_t<T>> &&
    !std::same_as<std::remove_cvref_t<T>, bool> &&
    !stratax::core::concept_detail::CharacterLike<T>;

template<typename T>
concept Numeric =
    Integral<T> ||
    std::floating_point<std::remove_cvref_t<T>> ||
    stratax::core::concept_detail::SupportedComplex<T>;

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
