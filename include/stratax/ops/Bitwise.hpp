// TODO: Validate shift counts to prevent invalid or undefined shifts.

// TODO: Add broadcasting support for array-array bitwise operations.

// TODO: Support mixed array/container and integral types once
// result-container and dtype promotion rules are defined.

#pragma once

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/ops/Broadcasting.hpp>

#include <functional>

#include <limits>
#include <type_traits>

template<typename Left, typename Shift>
requires Integral<Left> && Integral<Shift>
void validate_shift(Shift shift)
{
    using PromotedLeft = decltype(+std::declval<Left>());

    if constexpr (std::is_signed_v<Shift>)
    {
        if (shift < 0)
        {
            throw Exceptions::StrataxError(
                "Shift count cannot be negative.");
        }
    }

    using UnsignedShift = std::make_unsigned_t<Shift>;
    const auto count = static_cast<UnsignedShift>(shift);

    constexpr std::size_t width =
        std::numeric_limits<std::make_unsigned_t<PromotedLeft>>::digits;

    if (count >= width)
    {
        throw Exceptions::StrataxError(
            "Shift count exceeds the bit width of the operand.");
    }
}

/**
 * @brief Applies a callable element-wise to two same-shaped integer arrays.
 * @tparam A Common integral-valued Stratax array and result type.
 * @tparam Op Binary bitwise or shift callable.
 * @param lhs Left array operand.
 * @param rhs Right array operand.
 * @param op Callable invoked in flat iterator order.
 * @return Owning array with the same shape and converted operation results.
 * @throws Exceptions::ShapeError If the operand shapes differ.
 * @throws Any exception propagated by result allocation or @p op.
 * @complexity O(lhs.size() + lhs.rank()).
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
A binary_scalar_bitwise_op(const A& lhs, const Scalar& rhs, Op op)
{
	A result(lhs.shape());

	auto out = result.begin();
	for (auto it = lhs.begin(); it != lhs.end(); ++it, ++out)
	{
		*out = static_cast<typename A::value_type>(op(*it, rhs));
	}

	return result;
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
A binary_scalar_bitwise_op(const Scalar& lhs, const A& rhs, Op op)
{
	A result(rhs.shape());

	auto out = result.begin();
	for (auto it = rhs.begin(); it != rhs.end(); ++it, ++out)
	{
		*out = static_cast<typename A::value_type>(op(lhs, *it));
	}

	return result;
}

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

template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
auto operator&(const L& lhs, const R& rhs)
{
	return binary_bitwise_op(lhs, rhs, std::bit_and<>{});
}

template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
auto operator|(const L& lhs, const R& rhs)
{
	return binary_bitwise_op(lhs, rhs, std::bit_or<>{});
}

template<Array L, Array R>
requires (
	Integral<typename L::value_type> &&
	Integral<typename R::value_type>
)
auto operator^(const L& lhs, const R& rhs)
{
	return binary_bitwise_op(lhs, rhs, std::bit_xor<>{});
}

/** @brief Applies element-wise left shift using per-element shift counts. */
template<Array A>
requires Integral<typename A::value_type>
A operator<<(const A& lhs, const A& rhs)
{
    return binary_bitwise_op(
        lhs,
        rhs,
        [](auto left, auto right)
        {
            validate_shift<decltype(left)>(right);
            return left << right;
        });
}

/** @brief Applies element-wise right shift using per-element shift counts. */
template<Array A>
requires Integral<typename A::value_type>
A operator>>(const A& lhs, const A& rhs)
{
    return binary_bitwise_op(
        lhs,
        rhs,
        [](auto left, auto right)
        {
            validate_shift<decltype(left)>(right);
            return left << right;
        });
}

// Array-scalar

/** @brief Applies bitwise AND between each array element and a scalar. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
A operator&(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_bitwise_op(lhs, rhs, std::bit_and<>{});
}

/** @brief Applies bitwise OR between each array element and a scalar. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
A operator|(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_bitwise_op(lhs, rhs, std::bit_or<>{});
}

/** @brief Applies bitwise XOR between each array element and a scalar. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
A operator^(const A& lhs, const Scalar& rhs)
{
	return binary_scalar_bitwise_op(lhs, rhs, std::bit_xor<>{});
}

/** @brief Left-shifts each array element by a scalar shift count. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
A operator<<(const A& lhs, const Scalar& rhs)
{
    return binary_scalar_bitwise_op(
        lhs,
        rhs,
        [](auto left, auto right)
        {
            validate_shift<decltype(left)>(right);
            return left << right;
        });
}

/** @brief Right-shifts each array element by a scalar shift count. */
template<Array A, Integral Scalar>
requires Integral<typename A::value_type>
A operator>>(const A& lhs, const Scalar& rhs)
{
    return binary_scalar_bitwise_op(
        lhs,
        rhs,
        [](auto left, auto right)
        {
            validate_shift<decltype(left)>(right);
            return left << right;
        });
}

// Scalar-array (reverse)

/** @brief Applies bitwise AND between a scalar and each array element. */
template<Integral Scalar, Array A>
requires Integral<typename A::value_type>
A operator&(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_bitwise_op(lhs, rhs, std::bit_and<>{});
}

/** @brief Applies bitwise OR between a scalar and each array element. */
template<Integral Scalar, Array A>
requires Integral<typename A::value_type>
A operator|(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_bitwise_op(lhs, rhs, std::bit_or<>{});
}

/** @brief Applies bitwise XOR between a scalar and each array element. */
template<Integral Scalar, Array A>
requires Integral<typename A::value_type>
A operator^(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_bitwise_op(lhs, rhs, std::bit_xor<>{});
}

/** @brief Left-shifts a scalar value by each array element as shift count. */
template<Integral Scalar, Array A>
requires Integral<typename A::value_type>
A operator<<(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_bitwise_op(lhs, rhs, [](auto left, auto right) { return left << right; });
}

/** @brief Right-shifts a scalar value by each array element as shift count. */
template<Integral Scalar, Array A>
requires Integral<typename A::value_type>
A operator>>(const Scalar& lhs, const A& rhs)
{
	return binary_scalar_bitwise_op(lhs, rhs, [](auto left, auto right) { return left >> right; });
}

// In-place array-array

/** @brief Applies element-wise bitwise AND assignment. */
template<Array A>
requires Integral<typename A::value_type>
A& operator&=(A& lhs, const A& rhs)
{
	lhs = lhs & rhs;
	return lhs;
}

/** @brief Applies element-wise bitwise OR assignment. */
template<Array A>
requires Integral<typename A::value_type>
A& operator|=(A& lhs, const A& rhs)
{
	lhs = lhs | rhs;
	return lhs;
}

/** @brief Applies element-wise bitwise XOR assignment. */
template<Array A>
requires Integral<typename A::value_type>
A& operator^=(A& lhs, const A& rhs)
{
	lhs = lhs ^ rhs;
	return lhs;
}

/** @brief Applies element-wise left-shift assignment. */
template<Array A>
requires Integral<typename A::value_type>
A& operator<<=(A& lhs, const A& rhs)
{
	lhs = lhs << rhs;
	return lhs;
}

/** @brief Applies element-wise right-shift assignment. */
template<Array A>
requires Integral<typename A::value_type>
A& operator>>=(A& lhs, const A& rhs)
{
	lhs = lhs >> rhs;
	return lhs;
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
