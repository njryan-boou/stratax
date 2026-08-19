#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

#include <stratax/core/DTypes/DTypeTraits.hpp>

namespace stratax::core {

/**
 * @brief Determines the promoted dtype for two Stratax dtypes.
 *
 * The primary template is intentionally undefined. A promotion is only
 * available when a matching specialization exists.
 *
 * @tparam L Left operand dtype.
 * @tparam R Right operand dtype.
 */
template<typename L, typename R>
struct Promote;

/**
 * @brief Preserves a dtype when both operands have the same supported dtype.
 */
template<typename T>
requires requires
{
	typename DTypeTraits<T>::type;
	DTypeTraits<T>::kind;
}
struct Promote<T, T>
{
	using type = T;
};

/**
 * @brief Promotes two signed integer dtypes to the wider dtype.
 */
template<typename L, typename R>
requires (
	DTypeTraits<L>::kind == DTypeKind::SignedInteger &&
	DTypeTraits<R>::kind == DTypeKind::SignedInteger
)
struct Promote<L, R>
{
	using type = std::conditional_t<
		(DTypeTraits<L>::bits >= DTypeTraits<R>::bits),
		L,
		R
	>;
};

/**
 * @brief Promotes two unsigned integer dtypes to the wider dtype.
 */
template<typename L, typename R>
requires (
	DTypeTraits<L>::kind == DTypeKind::UnsignedInteger &&
	DTypeTraits<R>::kind == DTypeKind::UnsignedInteger
)
struct Promote<L, R>
{
	using type = std::conditional_t<
		(DTypeTraits<L>::bits >= DTypeTraits<R>::bits),
		L,
		R
	>;
};

/**
 * @brief Promotes mixed signed and unsigned integer dtypes.
 *
 * The smallest signed integer dtype capable of representing the complete
 * ranges of both operands is selected. If no supported signed integer dtype
 * is sufficiently wide, the result is float64.
 */
template<typename L, typename R>
requires (
	(DTypeTraits<L>::kind == DTypeKind::SignedInteger &&
	 DTypeTraits<R>::kind == DTypeKind::UnsignedInteger) ||
	(DTypeTraits<L>::kind == DTypeKind::UnsignedInteger &&
	 DTypeTraits<R>::kind == DTypeKind::SignedInteger)
)
struct Promote<L, R>
{
private:
	using signed_type = std::conditional_t<
		DTypeTraits<L>::kind == DTypeKind::SignedInteger,
		L,
		R
	>;

	using unsigned_type = std::conditional_t<
		DTypeTraits<L>::kind == DTypeKind::UnsignedInteger,
		L,
		R
	>;

	static constexpr std::size_t signed_bits =
		DTypeTraits<signed_type>::bits;

	static constexpr std::size_t unsigned_bits =
		DTypeTraits<unsigned_type>::bits;

	/*
	 * An N-bit unsigned integer requires N+1 bits in a signed
	 * representation to preserve its complete range.
	 */
	static constexpr std::size_t required_bits =
		signed_bits > unsigned_bits
			? signed_bits
			: unsigned_bits + 1;

public:
	using type = std::conditional_t<
		(required_bits <= 8),
		dtype::int8,
		std::conditional_t<
			(required_bits <= 16),
			dtype::int16,
			std::conditional_t<
				(required_bits <= 32),
				dtype::int32,
				std::conditional_t<
					(required_bits <= 64),
					dtype::int64,
					dtype::float64
				>
			>
		>
	>;
};

/**
 * @brief Promotes an integer dtype and a floating-point dtype.
 *
 * The floating dtype is preserved when it has sufficient significand
 * precision for the integer dtype. float32 is promoted to float64 when
 * additional precision is required.
 */
template<typename L, typename R>
requires (
	(
		(DTypeTraits<L>::kind == DTypeKind::SignedInteger ||
		 DTypeTraits<L>::kind == DTypeKind::UnsignedInteger) &&
		DTypeTraits<R>::kind == DTypeKind::Floating
	) ||
	(
		DTypeTraits<L>::kind == DTypeKind::Floating &&
		(DTypeTraits<R>::kind == DTypeKind::SignedInteger ||
		 DTypeTraits<R>::kind == DTypeKind::UnsignedInteger)
	)
)
struct Promote<L, R>
{
private:
	using integer_type = std::conditional_t<
		DTypeTraits<L>::kind == DTypeKind::SignedInteger ||
		DTypeTraits<L>::kind == DTypeKind::UnsignedInteger,
		L,
		R
	>;

	using floating_type = std::conditional_t<
		DTypeTraits<L>::kind == DTypeKind::Floating,
		L,
		R
	>;

	static constexpr int integer_digits =
		DTypeTraits<integer_type>::digits;

	static constexpr int floating_digits =
		DTypeTraits<floating_type>::digits;

public:
	using type = std::conditional_t<
		(integer_digits <= floating_digits),
		floating_type,
		std::conditional_t<
			std::same_as<floating_type, dtype::float32>,
			dtype::float64,
			floating_type
		>
	>;
};

/**
 * @brief Promotes two floating-point dtypes to the dtype with greater
 * numerical precision.
 */
template<typename L, typename R>
requires (
	DTypeTraits<L>::kind == DTypeKind::Floating &&
	DTypeTraits<R>::kind == DTypeKind::Floating
)
struct Promote<L, R>
{
	using type = std::conditional_t<
		(DTypeTraits<L>::rank >= DTypeTraits<R>::rank),
		L,
		R
	>;
};

/**
 * @brief Promotes two complex dtypes according to their real components.
 */
template<typename L, typename R>
requires (
	DTypeTraits<L>::kind == DTypeKind::Complex &&
	DTypeTraits<R>::kind == DTypeKind::Complex
)
struct Promote<L, R>
{
private:
	using left_component =
		complex_component_t<L>;

	using right_component =
		complex_component_t<R>;

	using promoted_component =
		typename Promote<left_component, right_component>::type;

public:
	using type =
		complex_from_real_t<promoted_component>;
};

/**
 * @brief Promotes a real numeric dtype and a complex dtype.
 *
 * The real dtype is promoted against the real component of the complex
 * dtype. The resulting real dtype is then mapped back to its corresponding
 * complex dtype.
 */
template<typename L, typename R>
requires (
	(
		(DTypeTraits<L>::kind == DTypeKind::SignedInteger ||
		 DTypeTraits<L>::kind == DTypeKind::UnsignedInteger ||
		 DTypeTraits<L>::kind == DTypeKind::Floating) &&
		DTypeTraits<R>::kind == DTypeKind::Complex
	) ||
	(
		DTypeTraits<L>::kind == DTypeKind::Complex &&
		(DTypeTraits<R>::kind == DTypeKind::SignedInteger ||
		 DTypeTraits<R>::kind == DTypeKind::UnsignedInteger ||
		 DTypeTraits<R>::kind == DTypeKind::Floating)
	)
)
struct Promote<L, R>
{
private:
	using real_type = std::conditional_t<
		DTypeTraits<L>::kind == DTypeKind::Complex,
		R,
		L
	>;

	using complex_type = std::conditional_t<
		DTypeTraits<L>::kind == DTypeKind::Complex,
		L,
		R
	>;

	using component_type =
		complex_component_t<complex_type>;

	using promoted_component =
		typename Promote<real_type, component_type>::type;

public:
	using type =
		complex_from_real_t<promoted_component>;
};

/**
 * @brief Convenience alias for the promoted dtype of two types.
 *
 * cv- and reference qualifiers are removed before promotion.
 */
template<typename L, typename R>
using promote_t = typename Promote<
	std::remove_cvref_t<L>,
	std::remove_cvref_t<R>
>::type;

} // namespace stratax::core
