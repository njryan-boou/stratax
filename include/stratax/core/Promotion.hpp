#pragma once

#include <type_traits>
#include <concepts>

#include <stratax/core/DTypeTraits.hpp>

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
 * @brief Preserves a dtype when both operands have the same type.
 */
template<typename T>
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
 * ranges of both input dtypes is selected. If no supported signed integer
 * dtype is wide enough, the result is float64.
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
	 * An N-bit unsigned integer needs N+1 bits in a signed representation
	 * in order to preserve its complete non-negative range.
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

template<typename L, typename R>
requires (
	((DTypeTraits<L>::kind == DTypeKind::SignedInteger ||
	  DTypeTraits<L>::kind == DTypeKind::UnsignedInteger) &&
	 DTypeTraits<R>::kind == DTypeKind::Floating) ||
	(DTypeTraits<L>::kind == DTypeKind::Floating &&
	 (DTypeTraits<R>::kind == DTypeKind::SignedInteger ||
	  DTypeTraits<R>::kind == DTypeKind::UnsignedInteger))
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