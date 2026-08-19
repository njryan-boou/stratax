#pragma once

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/ops/Broadcasting.hpp>

#include <functional>
#include <climits>
#include <cstdint>
#include <type_traits>

namespace stratax::core::bitwise_detail {

/**
 * @brief Tests whether a shift count is valid for a value type.
 * @tparam Value Type of the value being shifted.
 * @tparam Count Integral shift-count type.
 * @param count Shift count to validate.
 * @return `true` when @p count is non-negative and smaller than the bit width
 *         of `Value`; otherwise `false`.
 * @complexity O(1).
 * @internal
 */
template<typename Value, Integral Count>
constexpr bool valid_shift_count(const Count& count) noexcept
{
	if constexpr (std::is_signed_v<std::remove_cvref_t<Count>>)
	{
		if (count < 0)
		{
			return false;
		}
	}

	using value_type = std::remove_cvref_t<Value>;

	return static_cast<std::uintmax_t>(count) <
		sizeof(value_type) * CHAR_BIT;
}

/**
 * @brief Rejects a shift count that would produce undefined behavior.
 * @throws Exceptions::StrataxError If @p count is negative or is not smaller
 *         than the bit width of `Value`.
 * @complexity O(1).
 * @internal
 */
template<typename Value, Integral Count>
void require_valid_shift_count(const Count& count)
{
	if (!valid_shift_count<Value>(count))
	{
		throw Exceptions::StrataxError(
			"Shift count must be non-negative and less than the bit width of the shifted value.");
	}
}

/**
 * @brief Applies a validated scalar shift to every array element.
 * @return Owning array with the same container, dtype, and shape as @p lhs.
 * @throws Exceptions::StrataxError If @p rhs is not a valid shift count.
 * @complexity O(lhs.size()).
 * @internal
 */
template<Array A, Integral Count, typename Op>
requires Integral<typename A::value_type>
auto shift_scalar_op(
	const A& lhs,
	const Count& rhs,
	Op op)
{
	using value_type = typename A::value_type;

	require_valid_shift_count<value_type>(rhs);

	using result_type =
		stratax::core::rebind_array_t<A, value_type>;

	result_type result(lhs.shape());

	auto out = result.begin();

	for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
	{
		*out = static_cast<value_type>(
			op(*it, rhs));
	}

	return result;
}

/**
 * @brief Applies broadcasted per-element shift counts to an array.
 * @return Owning promoted container with the left operand's dtype and the
 *         common broadcasted shape.
 * @throws Exceptions::BroadcastError If the shapes are incompatible.
 * @throws Exceptions::StrataxError If any used shift count is invalid.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
 * @internal
 */
template<Array L, Array R, typename Op>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
auto shift_array_op(
	const L& lhs,
	const R& rhs,
	Op op)
{
	using value_type = typename L::value_type;

	using result_type =
		stratax::core::promote_array_t<
			L,
			R,
			value_type>;

	const auto result_shape =
		broadcasted_shape(lhs.shape(), rhs.shape());

	result_type result(result_shape);

	for (std::size_t i = 0; i < result.size(); ++i)
	{
		const std::size_t lhs_index =
			stratax::core::broadcast_detail::flat_operand_index(
				i,
				result_shape,
				lhs.shape());

		const std::size_t rhs_index =
			stratax::core::broadcast_detail::flat_operand_index(
				i,
				result_shape,
				rhs.shape());

		const auto count = rhs[rhs_index];

		require_valid_shift_count<value_type>(count);

		result[i] = static_cast<value_type>(
			op(lhs[lhs_index], count));
	}

	return result;
}

/**
 * @brief Shifts a scalar by every count stored in an array.
 * @return Owning array with @p rhs's container and shape and the scalar dtype.
 * @throws Exceptions::StrataxError If any shift count is invalid.
 * @complexity O(rhs.size()).
 * @internal
 */
template<Integral Scalar, Array A, typename Op>
requires Integral<typename A::value_type>
auto scalar_shift_array_op(
	const Scalar& lhs,
	const A& rhs,
	Op op)
{
	using value_type = std::remove_cvref_t<Scalar>;

	using result_type =
		stratax::core::rebind_array_t<
			A,
			value_type>;

	result_type result(rhs.shape());

	auto out = result.begin();

	for (auto it = rhs.begin(); it != rhs.end(); ++it, ++out)
	{
		require_valid_shift_count<value_type>(*it);

		*out = static_cast<value_type>(
			op(lhs, *it));
	}

	return result;
}

/**
 * @brief Applies a broadcasted bitwise operation to @p lhs in place.
 * @return Reference to @p lhs, whose shape and dtype remain unchanged.
 * @throws Exceptions::BroadcastError If broadcasting is impossible or would
 *         change the shape of @p lhs.
 * @complexity O(lhs.size() * lhs.rank()).
 * @internal
 */
template<Array L, Array R, typename Op>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
L& compound_bitwise_op(
	L& lhs,
	const R& rhs,
	Op op)
{
	const auto result_shape =
		broadcasted_shape(lhs.shape(), rhs.shape());

	if (result_shape != lhs.shape())
	{
		throw Exceptions::BroadcastError(
			"Compound bitwise assignment cannot change the left-hand shape.");
	}

	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		const std::size_t rhs_index =
			stratax::core::broadcast_detail::flat_operand_index(
				i,
				lhs.shape(),
				rhs.shape());

		lhs[i] = static_cast<typename L::value_type>(
			op(lhs[i], rhs[rhs_index]));
	}

	return lhs;
}

/**
 * @brief Applies a scalar bitwise operation to @p lhs in place.
 * @return Reference to @p lhs.
 * @complexity O(lhs.size()).
 * @internal
 */
template<Array A, Integral Scalar, typename Op>
requires Integral<typename A::value_type>
A& compound_scalar_bitwise_op(
	A& lhs,
	const Scalar& rhs,
	Op op)
{
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		lhs[i] = static_cast<typename A::value_type>(
			op(lhs[i], rhs));
	}

	return lhs;
}

/**
 * @brief Applies validated broadcasted shift counts to @p lhs in place.
 *
 * All used counts are validated before mutation, so an invalid count leaves
 * @p lhs unchanged.
 *
 * @return Reference to @p lhs, whose shape and dtype remain unchanged.
 * @throws Exceptions::BroadcastError If broadcasting is impossible or would
 *         change the shape of @p lhs.
 * @throws Exceptions::StrataxError If any used shift count is invalid.
 * @complexity O(lhs.size() * lhs.rank()).
 * @internal
 */
template<Array L, Array R, typename Op>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
L& compound_shift_op(
	L& lhs,
	const R& rhs,
	Op op)
{
	const auto result_shape =
		broadcasted_shape(lhs.shape(), rhs.shape());

	if (result_shape != lhs.shape())
	{
		throw Exceptions::BroadcastError(
			"Compound shift assignment cannot change the left-hand shape.");
	}

	using value_type = typename L::value_type;

	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		const std::size_t rhs_index =
			stratax::core::broadcast_detail::flat_operand_index(
				i,
				lhs.shape(),
				rhs.shape());

		require_valid_shift_count<value_type>(rhs[rhs_index]);
	}

	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		const std::size_t rhs_index =
			stratax::core::broadcast_detail::flat_operand_index(
				i,
				lhs.shape(),
				rhs.shape());

		const auto count = rhs[rhs_index];

		lhs[i] = static_cast<value_type>(
			op(lhs[i], count));
	}

	return lhs;
}

/**
 * @brief Applies a bitwise callable to two broadcast-compatible integer arrays.
 *
 * Matching container families are preserved and mixed families promote to
 * Tensor. The result uses the promoted dtype of the operands.
 *
 * @tparam L Left integral-valued Stratax array type.
 * @tparam R Right integral-valued Stratax array type.
 * @tparam Op Binary bitwise callable.
 * @param lhs Left array operand.
 * @param rhs Right array operand.
 * @param op Callable invoked in flat iterator order.
 * @return Owning promoted array with the common broadcasted shape.
 * @throws Exceptions::BroadcastError If the operand shapes are incompatible.
 * @throws Any exception propagated by result allocation or @p op.
 * @complexity O(n * r), where `n` is result size and `r` is result rank.
 */
template<Array L, Array R, typename Op>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
auto binary_bitwise_op(
	const L& lhs,
	const R& rhs,
	Op op)
{
	return broadcasted_op(lhs, rhs, op);
}

/**
 * @brief Applies a callable to every integer array element and a right scalar.
 * @tparam A Integral-valued Stratax array and result type.
 * @tparam Scalar Integral scalar type.
 * @tparam Op Binary bitwise or shift callable.
 * @param lhs Array supplying every left argument.
 * @param rhs Scalar supplied as every right argument.
 * @param op Callable invoked in flat iterator order.
 * @return Owning array with the same shape as @p lhs.
 * @throws Any exception propagated by result allocation or @p op.
 * @complexity O(lhs.size()).
 */
template<Array A, Integral Scalar, typename Op>
requires Integral<typename A::value_type>
auto binary_scalar_bitwise_op(
	const A& lhs,
	const Scalar& rhs,
	Op op)
{
	return broadcasted_op(lhs, rhs, op);
}

/**
 * @brief Applies a callable to a left scalar and every integer array element.
 * @tparam Scalar Integral scalar type.
 * @tparam A Integral-valued Stratax array and result type.
 * @tparam Op Binary bitwise or shift callable.
 * @param lhs Scalar supplied as every left argument.
 * @param rhs Array supplying every right argument.
 * @param op Callable invoked in flat iterator order.
 * @return Owning array with the same shape as @p rhs.
 * @throws Any exception propagated by result allocation or @p op.
 * @complexity O(rhs.size()).
 */
template<Integral Scalar, Array A, typename Op>
requires Integral<typename A::value_type>
auto binary_scalar_bitwise_op(
	const Scalar& lhs,
	const A& rhs,
	Op op)
{
	return broadcasted_op(lhs, rhs, op);
}

} // namespace stratax::core::bitwise_detail

// Unary

/**
 * @brief Applies bitwise NOT to each element of an integer array.
 * @param value Source array.
 * @return Owning array with the same shape containing `~value[i]`.
 * @complexity O(value.size()).
 */
template<Array A>
requires Integral<typename A::value_type>
A operator~(const A& value)
{
	A result(value.shape());

	auto out = result.begin();
	for (auto it = value.begin(); it != value.end(); ++it, ++out)
	{
		*out = static_cast<typename A::value_type>(std::bit_not<>{}(*it));
	}

	return result;
}

// Array-array

/** @brief Computes the broadcasted element-wise bitwise AND. @return Owning promoted array with the common broadcasted shape. @throws Exceptions::BroadcastError If the shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
auto operator&(const L& lhs, const R& rhs)
{
	return stratax::core::bitwise_detail::binary_bitwise_op(
		lhs, rhs, std::bit_and<>{});
}

/** @brief Computes the broadcasted element-wise bitwise OR. @return Owning promoted array with the common broadcasted shape. @throws Exceptions::BroadcastError If the shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
auto operator|(const L& lhs, const R& rhs)
{
	return stratax::core::bitwise_detail::binary_bitwise_op(
		lhs, rhs, std::bit_or<>{});
}

/** @brief Computes the broadcasted element-wise bitwise XOR. @return Owning promoted array with the common broadcasted shape. @throws Exceptions::BroadcastError If the shapes are incompatible. @complexity O(n * r). */
template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
auto operator^(const L& lhs, const R& rhs)
{
	return stratax::core::bitwise_detail::binary_bitwise_op(
		lhs, rhs, std::bit_xor<>{});
}

/** @brief Left-shifts broadcasted left values by corresponding right counts. @return Owning array with the promoted container and left dtype. @throws Exceptions::BroadcastError If the shapes are incompatible. @throws Exceptions::StrataxError If a used count is invalid. @complexity O(n * r). */
template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
auto operator<<(const L& lhs, const R& rhs)
{
	return stratax::core::bitwise_detail::shift_array_op(
		lhs,
		rhs,
		[](auto value, auto count)
		{
			return value << count;
		});
}

/** @brief Right-shifts broadcasted left values by corresponding right counts. @return Owning array with the promoted container and left dtype. @throws Exceptions::BroadcastError If the shapes are incompatible. @throws Exceptions::StrataxError If a used count is invalid. @complexity O(n * r). */
template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
auto operator>>(const L& lhs, const R& rhs)
{
	return stratax::core::bitwise_detail::shift_array_op(
		lhs,
		rhs,
		[](auto value, auto count)
		{
			return value >> count;
		});
}

// Array-scalar

/** @brief Applies bitwise AND between each array element and a scalar. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
auto operator&(const A& lhs, const Scalar& rhs)
{
	return stratax::core::bitwise_detail::binary_scalar_bitwise_op(
		lhs, rhs, std::bit_and<>{});
}

/** @brief Applies bitwise OR between each array element and a scalar. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
auto operator|(const A& lhs, const Scalar& rhs)
{
	return stratax::core::bitwise_detail::binary_scalar_bitwise_op(
		lhs, rhs, std::bit_or<>{});
}

/** @brief Applies bitwise XOR between each array element and a scalar. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
auto operator^(const A& lhs, const Scalar& rhs)
{
	return stratax::core::bitwise_detail::binary_scalar_bitwise_op(
		lhs, rhs, std::bit_xor<>{});
}

/** @brief Left-shifts every array element by a validated scalar count. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
auto operator<<(const A& lhs, const Scalar& rhs)
{
	return stratax::core::bitwise_detail::shift_scalar_op(
		lhs,
		rhs,
		[](auto value, auto count)
		{
			return value << count;
		});
}

/** @brief Right-shifts every array element by a validated scalar count. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
auto operator>>(const A& lhs, const Scalar& rhs)
{
	return stratax::core::bitwise_detail::shift_scalar_op(
		lhs,
		rhs,
		[](auto value, auto count)
		{
			return value >> count;
		});
}

// Scalar-array (reverse)

/** @brief Applies bitwise AND between a scalar and each array element. */
/** @brief Left-shifts a scalar by every validated count in an array. @return Owning result with the array container and scalar dtype. @throws Exceptions::StrataxError If any count is invalid. @complexity O(rhs.size()). */
template<Integral Scalar, Array A>
requires Integral<typename A::value_type>
auto operator&(const Scalar& lhs, const A& rhs)
{
	return stratax::core::bitwise_detail::binary_scalar_bitwise_op(
		lhs, rhs, std::bit_and<>{});
}

/** @brief Applies bitwise OR between a scalar and each array element. */
/** @brief Right-shifts a scalar by every validated count in an array. @return Owning result with the array container and scalar dtype. @throws Exceptions::StrataxError If any count is invalid. @complexity O(rhs.size()). */
template<Integral Scalar, Array A>
requires Integral<typename A::value_type>
auto operator|(const Scalar& lhs, const A& rhs)
{
	return stratax::core::bitwise_detail::binary_scalar_bitwise_op(
		lhs, rhs, std::bit_or<>{});
}

/** @brief Applies bitwise XOR between a scalar and each array element. */
template<Integral Scalar, Array A>
requires Integral<typename A::value_type>
auto operator^(const Scalar& lhs, const A& rhs)
{
	return stratax::core::bitwise_detail::binary_scalar_bitwise_op(
		lhs, rhs, std::bit_xor<>{});
}

template<Integral Scalar, Array A>
requires Integral<typename A::value_type>
auto operator<<(const Scalar& lhs, const A& rhs)
{
	return stratax::core::bitwise_detail::scalar_shift_array_op(
		lhs,
		rhs,
		[](auto value, auto count)
		{
			return value << count;
		});
}

template<Integral Scalar, Array A>
requires Integral<typename A::value_type>
auto operator>>(const Scalar& lhs, const A& rhs)
{
	return stratax::core::bitwise_detail::scalar_shift_array_op(
		lhs,
		rhs,
		[](auto value, auto count)
		{
			return value >> count;
		});
}

// In-place array-array

/** @brief Applies broadcasted bitwise AND assignment. @return Reference to @p lhs. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the left shape. @complexity O(lhs.size() * lhs.rank()). */
template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
L& operator&=(L& lhs, const R& rhs)
{
	return stratax::core::bitwise_detail::compound_bitwise_op(
		lhs, rhs, std::bit_and<>{});
}

/** @brief Applies broadcasted bitwise OR assignment. @return Reference to @p lhs. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the left shape. @complexity O(lhs.size() * lhs.rank()). */
template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
L& operator|=(L& lhs, const R& rhs)
{
	return stratax::core::bitwise_detail::compound_bitwise_op(
		lhs, rhs, std::bit_or<>{});
}

/** @brief Applies broadcasted bitwise XOR assignment. @return Reference to @p lhs. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the left shape. @complexity O(lhs.size() * lhs.rank()). */
template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
L& operator^=(L& lhs, const R& rhs)
{
	return stratax::core::bitwise_detail::compound_bitwise_op(
		lhs, rhs, std::bit_xor<>{});
}

/** @brief Applies broadcasted left-shift assignment after validating every count. @return Reference to @p lhs. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the left shape. @throws Exceptions::StrataxError If a used count is invalid. @complexity O(lhs.size() * lhs.rank()). */
template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
L& operator<<=(L& lhs, const R& rhs)
{
	return stratax::core::bitwise_detail::compound_shift_op(
		lhs,
		rhs,
		[](auto value, auto count) {
			return value << count;
		});
}

/** @brief Applies broadcasted right-shift assignment after validating every count. @return Reference to @p lhs. @throws Exceptions::BroadcastError If broadcasting is impossible or would change the left shape. @throws Exceptions::StrataxError If a used count is invalid. @complexity O(lhs.size() * lhs.rank()). */
template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
L& operator>>=(L& lhs, const R& rhs)
{
	return stratax::core::bitwise_detail::compound_shift_op(
		lhs,
		rhs,
		[](auto value, auto count) {
			return value >> count;
		});
}

// In-place array-scalar

/** @brief Applies bitwise AND assignment with a scalar. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
A& operator&=(A& lhs, const Scalar& rhs)
{
	lhs = lhs & rhs;
	return lhs;
}

/** @brief Applies bitwise OR assignment with a scalar. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
A& operator|=(A& lhs, const Scalar& rhs)
{
	lhs = lhs | rhs;
	return lhs;
}

/** @brief Applies bitwise XOR assignment with a scalar. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
A& operator^=(A& lhs, const Scalar& rhs)
{
	lhs = lhs ^ rhs;
	return lhs;
}

/** @brief Applies left-shift assignment with a scalar shift count. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
A& operator<<=(A& lhs, const Scalar& rhs)
{
	lhs = lhs << rhs;
	return lhs;
}

/** @brief Applies right-shift assignment with a scalar shift count. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
A& operator>>=(A& lhs, const Scalar& rhs)
{
	lhs = lhs >> rhs;
	return lhs;
}
