#pragma once

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/validation/Validation.hpp>

/** @brief Compares two array-like containers for exact equality. */
template<Array A>
[[nodiscard]] bool operator==(const A& lhs, const A& rhs)
{
	if (!stratax::core::validation::same_shape(lhs, rhs))
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

/** @brief Compares two array-like containers for inequality. */
template<Array A>
[[nodiscard]] bool operator!=(const A& lhs, const A& rhs)
{
	return !(lhs == rhs);
}
