// TODO: Add element-wise comparison operations when boolean array support is implemented.

#pragma once

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/exceptions/Exceptions.hpp>

/**
 * @brief Tests two arrays for exact shape-aware equality.
 *
 * Shapes are compared before values; broadcasting is never attempted. When the
 * shapes match, values are compared in flat iterator order using their exact
 * `operator==`. The scan stops at the first unequal element. Empty arrays are
 * equal only when their shapes are equal.
 *
 * @tparam A Common Stratax array type.
 * @param lhs Left array to compare.
 * @param rhs Right array to compare.
 * @return `true` when shapes and every corresponding value are equal.
 * @complexity O(r + n) in the worst case, where `r` is rank and `n` is size.
 */
template<Array A>
[[nodiscard]] bool operator==(const A& lhs, const A& rhs)
{
	if (lhs.shape() != rhs.shape())
	{
		return false;
	}

	auto it1 = lhs.begin();
	auto it2 = rhs.begin();
	const auto end = lhs.end();

	for (; it1 != end; ++it1, ++it2)
	{
		if (*it1 != *it2)
		{
			return false;
		}
	}
	return true;
}

/**
 * @brief Tests two arrays for shape or value inequality.
 * @tparam A Common Stratax array type.
 * @param lhs Left array to compare.
 * @param rhs Right array to compare.
 * @return `true` exactly when `operator==(lhs, rhs)` is false.
 * @complexity O(r + n) in the worst case, where `r` is rank and `n` is size.
 */
template<Array A>
[[nodiscard]] bool operator!=(const A& lhs, const A& rhs)
{
	return !(lhs == rhs);
}
