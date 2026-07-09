
#pragma once

#include <stratax/core/Concepts.hpp>

template<Array A>
bool operator==(const A& lhs, const A& rhs)
{
    if (lhs.size() != rhs.size() || lhs.shape() != rhs.shape())
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
bool operator!=(const A& lhs, const A& rhs)
{
    return !(lhs == rhs);
}
