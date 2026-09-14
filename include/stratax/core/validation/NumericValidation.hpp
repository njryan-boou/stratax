/** @file
 * @brief Shared validation for division operands and integral shift counts.
 */
#pragma once

#include <stratax/core/dtypes/Concepts.hpp>
#include <stratax/exceptions/Exceptions.hpp>

#include <climits>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace stratax::core::validation {

/**
 * @brief Rejects zero divisors and unrepresentable native signed division.
 *
 * Integral checks use the type of the native `lhs / rhs` expression, after
 * integer promotion and usual arithmetic conversions. Result-container dtype
 * promotion occurs later and does not change whether that expression is valid.
 * A narrow signed minimum divided by minus one can therefore be valid when
 * the expression evaluates in a wider signed type.
 *
 * @tparam L Numeric left operand type.
 * @tparam R Numeric right operand type.
 * @param lhs Original left element or scalar, before result dtype conversion.
 * @param rhs Original right element or scalar, before result dtype conversion.
 * @throws Exceptions::ZeroDivisionError If @p rhs equals zero, including
 *         floating-point signed zero or complex zero.
 * @throws Exceptions::OverflowError If integral division would evaluate the
 *         native signed expression type's minimum value divided by minus one.
 * @note This does not check narrowing to a result dtype or floating-point overflow.
 * @complexity O(1).
 * @internal
 */
template<Numeric L, Numeric R>
void require_valid_division(const L& lhs, const R& rhs)
{
	if (rhs == R{})
	{
		throw Exceptions::ZeroDivisionError("Division by zero.");
	}

	if constexpr (Integral<L> && Integral<R>)
	{
		using expression_type = decltype(lhs / rhs);
		if constexpr (std::is_signed_v<expression_type>)
		{
			if (static_cast<expression_type>(lhs) ==
				std::numeric_limits<expression_type>::min() &&
				static_cast<expression_type>(rhs) == expression_type{-1})
			{
				throw Exceptions::OverflowError("Signed integer division overflow.");
			}
		}
	}
}

/**
 * @brief Tests whether a shift count is valid for a value type.
 * @tparam Value Type of the value being shifted.
 * @tparam Count Integral shift-count type.
 * @param count Shift count to validate.
 * @return `true` when @p count is non-negative and smaller than the bit width
 *         of `Value`; otherwise `false`.
 * @invariant The shift count is never modified.
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
 * @brief Rejects shift counts outside the stored value type's bit width.
 * @invariant A successful return guarantees that @p count is valid for `Value`.
 * @throws Exceptions::ValueError If @p count is negative or is not smaller
 *         than the bit width of `Value`.
 * @complexity O(1).
 * @internal
 */
template<typename Value, Integral Count>
void require_valid_shift_count(const Count& count)
{
	if (!valid_shift_count<Value>(count))
	{
		throw Exceptions::ValueError("Invalid shift count.");
	}
}

} // namespace stratax::core::validation
