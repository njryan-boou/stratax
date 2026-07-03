
#pragma once


template<typename Array>
bool operator==(const Array& lhs, const Array& rhs)
{
    if (lhs.shape() != rhs.shape())
    {
        return false;
    }

    auto it1 = lhs.begin();
    auto it2 = rhs.begin();

    for (; it1 != lhs.end(); ++it1, ++it2)
    {
        if (*it1 != *it2)
        {
            return false;
        }
    }
    return true;
}

template <typename Array>
bool operator!=(const Array& lhs, const Array& rhs)
{
    return !(lhs == rhs);
}
