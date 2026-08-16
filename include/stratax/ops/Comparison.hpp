// TODO: Add element-wise comparison operations when boolean array support is implemented.

#pragma once

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/exceptions/Exceptions.hpp>

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

template<Array A>
[[nodiscard]] bool operator!=(const A& lhs, const A& rhs)
{
	return !(lhs == rhs);
}
