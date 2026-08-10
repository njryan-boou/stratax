#pragma once

#include <cstddef>
#include <limits>

#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::core::validation {

/** @brief Converts a signed index to `std::size_t` after rejecting negatives. */
inline std::size_t nonnegative_index(std::ptrdiff_t value, const char* message)
{
    if (value < 0)
    {
        throw Exceptions::IndexError(message);
    }

    return static_cast<std::size_t>(value);
}

/** @brief Normalizes a potentially negative index against a container size. */
inline std::size_t normalize_index(std::ptrdiff_t value, std::size_t size, const char* message)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    {
        throw Exceptions::IndexError(message);
    }

    const std::ptrdiff_t limit = static_cast<std::ptrdiff_t>(size);
    std::ptrdiff_t normalized = value;

    if (normalized < 0)
    {
        normalized += limit;
    }

    if (normalized < 0 || normalized >= limit)
    {
        throw Exceptions::IndexError(message);
    }

    return static_cast<std::size_t>(normalized);
}

/** @brief Requires an index to be less than a size. */
inline void require_index(std::size_t index, std::size_t size, const char* message)
{
    if (index >= size)
    {
        throw Exceptions::IndexError(message);
    }
}

/** @brief Requires a value to be less than or equal to an upper bound. */
inline void require_at_most(std::size_t value, std::size_t upper, const char* message)
{
    if (value > upper)
    {
        throw Exceptions::IndexError(message);
    }
}

}
