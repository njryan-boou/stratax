#pragma once

#include <cstddef>
#include <string>

#include <stratax/exceptions/Exceptions.hpp>

using size_type = std::size_t;
using difference_type = std::ptrdiff_t;

namespace stratax::indexing {

inline size_type normalize_index(
    difference_type index,
    size_type size
    )
{
    if (index >= 0)
    {
        const size_type normalized = static_cast<size_type>(index);

        if (normalized >= size)
        {
            throw Exceptions::IndexError("Index " + std::to_string(index) + 
                " is out of bounds for array with size " + std::to_string(size) + ".");
        }

        return normalized;
    }

    const size_type magnitude =
        static_cast<size_type>(-(index + 1)) + 1;

    if (magnitude > size)
    {
        throw Exceptions::IndexError("Index " + std::to_string(index) + 
            " is out of bounds for array with size " + std::to_string(size) + ".");
    }

    return size - magnitude;
}

} // namespace stratax::indexing