#pragma once

#include <concepts>
#include <complex>
#include <type_traits>

#include <stratax/core/dtypes/Types.hpp>

namespace stratax::core::concept_detail {

template<typename T, typename... Candidates>
concept SameAsAny =
	(std::same_as<std::remove_cvref_t<T>, Candidates> || ...);

template<typename T>
concept BoolLike =
	std::same_as<std::remove_cvref_t<T>, stratax::dtype::bool_>;

template<typename T>
concept SupportedIntegral =
	SameAsAny<
		T,
		stratax::dtype::int8,
		stratax::dtype::int16,
		stratax::dtype::int32,
		stratax::dtype::int64,
		stratax::dtype::uint8,
		stratax::dtype::uint16,
		stratax::dtype::uint32,
		stratax::dtype::uint64>;

template<typename T>
concept SupportedFloating =
	SameAsAny<
		T,
		stratax::dtype::float32,
		stratax::dtype::float64,
		stratax::dtype::longdouble>;

template<typename T>
concept SupportedComplex =
	SameAsAny<
		T,
		stratax::dtype::complex64,
		stratax::dtype::complex128,
		stratax::dtype::clongdouble>;

}

template<typename T>
concept Integral =
	stratax::core::concept_detail::SupportedIntegral<T>;

template<typename T>
concept Numeric =
	Integral<T> ||
	stratax::core::concept_detail::SupportedFloating<T> ||
	stratax::core::concept_detail::SupportedComplex<T>;

template<typename T>
concept DType =
	Numeric<T> ||
	stratax::core::concept_detail::BoolLike<T>;

namespace stratax::container {

/** @brief Forward declaration of the rank-one dtype array container. */
template<typename T>
requires DType<T>
class Vector;

/** @brief Forward declaration of the rank-two dtype array container. */
template<typename T>
requires DType<T>
class Matrix;

/** @brief Forward declaration of the arbitrary-rank dtype array container. */
template<typename T>
requires DType<T>
class Tensor;

} // namespace stratax::container

/**
 * @brief Trait identifying supported Stratax owning array specializations.
 *
 * The primary template is false. Specializations for Vector, Matrix, and
 * Tensor are true when their element type satisfies DType.
 *
 * @tparam T Type to inspect. Use Array when cv/ref normalization is desired.
 */
template<typename T>
struct is_array : std::false_type {};

/** @brief Marks dtype-compatible Vector specializations as Stratax arrays. */
template<typename T>
requires DType<T>
struct is_array<stratax::container::Vector<T>> : std::true_type {};

/** @brief Marks dtype-compatible Matrix specializations as Stratax arrays. */
template<typename T>
requires DType<T>
struct is_array<stratax::container::Matrix<T>> : std::true_type {};

/** @brief Marks dtype-compatible Tensor specializations as Stratax arrays. */
template<typename T>
requires DType<T>
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

template<typename T>
concept Ordered =
	DType<T> &&
	!stratax::core::concept_detail::SupportedComplex<T>;

template<typename T>
concept RealNumeric =
    Integral<T> ||
    std::floating_point<std::remove_cvref_t<T>>;
