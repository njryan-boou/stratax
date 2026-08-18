#pragma once

#include <concepts>
#include <complex>
#include <type_traits>

namespace stratax::core::concept_detail {

/**
 * @brief Tests whether a type matches any type in a candidate list.
 *
 * cv- and reference qualifiers are removed from @p T before comparison.
 *
 * @tparam T Type to classify.
 * @tparam Candidates Exact unqualified types accepted by the concept.
 * @internal
 */
template<typename T, typename... Candidates>
concept SameAsAny =
	(std::same_as<std::remove_cvref_t<T>, Candidates> || ...);

/**
 * @brief Identifies built-in character and code-unit types.
 *
 * Plain character and code-unit types are intentionally excluded from Integral
 * and Numeric so textual data is not accepted as numerical array storage.
 * `signed char` and `unsigned char` remain numeric because implementations
 * commonly use them for `std::int8_t` and `std::uint8_t`.
 *
 * @tparam T Type to classify after removing cv/ref qualifiers.
 * @internal
 */
template<typename T>
concept CharacterLike =
	SameAsAny<
		T,
		char,
		wchar_t,
		char8_t,
		char16_t,
		char32_t>;

/**
 * @brief Identifies bool after removing cv/ref qualifiers.
 * @tparam T Type to classify.
 * @internal
 */
template<typename T>
concept BoolLike =
	std::same_as<std::remove_cvref_t<T>, bool>;

/**
 * @brief Identifies supported standard complex specializations.
 *
 * Only complex numbers backed by `float`, `double`, or `long double` are
 * accepted.
 *
 * @tparam T Type to classify after removing cv/ref qualifiers.
 * @internal
 */
template<typename T>
concept SupportedComplex =
	SameAsAny<
		T,
		std::complex<float>,
		std::complex<double>,
		std::complex<long double>>;

} // namespace stratax::core::concept_detail

/**
 * @brief Identifies non-boolean, non-character integral scalar types.
 *
 * cv- and reference qualifiers are ignored. Standard signed and unsigned
 * integer types, including `signed char` and `unsigned char`, satisfy this
 * concept. `bool`, plain character/code-unit types, floating types, and complex
 * types do not.
 *
 * @tparam T Type to classify.
 */
template<typename T>
concept Integral =
	std::integral<std::remove_cvref_t<T>> &&
	!stratax::core::concept_detail::BoolLike<T> &&
	!stratax::core::concept_detail::CharacterLike<T>;

/**
 * @brief Identifies scalar types supported by Stratax containers and ops.
 *
 * Numeric accepts Integral types, built-in floating-point types, and
 * `std::complex<float>`, `std::complex<double>`, or
 * `std::complex<long double>`. cv- and reference qualifiers are ignored.
 *
 * @tparam T Type to classify.
 */
template<typename T>
concept Numeric =
	Integral<T> ||
	std::floating_point<std::remove_cvref_t<T>> ||
	stratax::core::concept_detail::SupportedComplex<T>;

namespace stratax::container {

/** @brief Forward declaration of the rank-one numeric array container. */
template<typename T>
requires Numeric<T>
class Vector;

/** @brief Forward declaration of the rank-two numeric array container. */
template<typename T>
requires Numeric<T>
class Matrix;

/** @brief Forward declaration of the arbitrary-rank numeric array container. */
template<typename T>
requires Numeric<T>
class Tensor;

} // namespace stratax::container

/**
 * @brief Trait identifying supported Stratax owning array specializations.
 *
 * The primary template is false. Specializations for Vector, Matrix, and
 * Tensor are true when their element type satisfies Numeric.
 *
 * @tparam T Type to inspect. Use Array when cv/ref normalization is desired.
 */
template<typename T>
struct is_array : std::false_type {};

/** @brief Marks numeric Vector specializations as Stratax arrays. */
template<typename T>
requires Numeric<T>
struct is_array<stratax::container::Vector<T>> : std::true_type {};

/** @brief Marks numeric Matrix specializations as Stratax arrays. */
template<typename T>
requires Numeric<T>
struct is_array<stratax::container::Matrix<T>> : std::true_type {};

/** @brief Marks numeric Tensor specializations as Stratax arrays. */
template<typename T>
requires Numeric<T>
struct is_array<stratax::container::Tensor<T>> : std::true_type {};

/**
 * @brief Identifies supported Stratax owning array types.
 *
 * Array accepts Vector, Matrix, and Tensor specializations and ignores cv- and
 * reference qualifiers. It is intentionally nominal rather than structural;
 * unrelated types with a similar interface do not satisfy it.
 *
 * @tparam T Type to classify.
 */
template<typename T>
concept Array =
	is_array<std::remove_cvref_t<T>>::value;
