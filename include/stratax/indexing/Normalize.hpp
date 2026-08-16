#pragma once

#include <cstddef>
#include <string>

#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::indexing {

inline std::size_t normalize_index(
    std::ptrdiff_t index,
    std::size_t size)
{
    if (index >= 0)
    {
        const std::size_t normalized = static_cast<std::size_t>(index);

        if (normalized >= size)
        {
            throw Exceptions::IndexError("Index " + std::to_string(index) + 
                " is out of bounds for array with size " + std::to_string(size) + ".");
        }

        return normalized;
    }

    const std::size_t magnitude =
        static_cast<std::size_t>(-(index + 1)) + 1;

    if (magnitude > size)
    {
        throw Exceptions::IndexError("Index " + std::to_string(index) + 
            " is out of bounds for array with size " + std::to_string(size) + ".");
    }

    return size - magnitude;
}

} // namespace stratax::indexing