/** @file
 * @brief Validation of explicitly checked numeric operations.
 */
#pragma once

#include <stratax/core/dtypes/Concepts.hpp>
#include <stratax/exceptions/Exceptions.hpp>

#include <limits>
#include <type_traits>

namespace stratax::core::numeric_detail {

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

} // namespace stratax::core::numeric_detail
