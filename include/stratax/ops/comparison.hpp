
#pragma once

#include <stratax/core/Concepts.hpp>
#include <stratax/core/Validation.hpp>

/**
 * @brief Compares two array-like containers for exact equality.
 *
 * @tparam A Array type satisfying the Stratax array concept.
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 *
 * @return `true` if both arrays have the same shape and element values.
 */
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

/**
 * @brief Compares two array-like containers for inequality.
 *
 * @tparam A Array type satisfying the Stratax array concept.
 * @param lhs Left-hand operand.
 * @param rhs Right-hand operand.
 *
 * @return `true` if the arrays differ in shape or element values.
 */
template<Array A>
[[nodiscard]] bool operator!=(const A& lhs, const A& rhs)
{
    return !(lhs == rhs);
}
