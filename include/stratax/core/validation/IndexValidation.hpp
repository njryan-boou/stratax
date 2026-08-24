#pragma once

#include <cstddef>

#include <stratax/exceptions/IndexErrors.hpp>

namespace stratax::validation {

using size_type = std::size_t;
using difference_type = std::ptrdiff_t;

namespace detail {

[[nodiscard]]
constexpr size_type negative_magnitude(
    difference_type index) noexcept
{
    // Avoids overflowing when index == PTRDIFF_MIN.
    return static_cast<size_type>(-(index + 1)) + 1;
}

} // namespace detail

[[nodiscard]]
constexpr bool valid_index(
    difference_type index,
    size_type size) noexcept
{
    if (index >= 0)
    {
        return static_cast<size_type>(index) < size;
    }

    return detail::negative_magnitude(index) <= size;
}

inline void require_valid_index(
    difference_type index,
    size_type size)
{
    if (!valid_index(index, size))
    {
        throw Exceptions::IndexError(index, size);
    }
}

[[nodiscard]]
inline size_type normalize_index(
    difference_type index,
    size_type size)
{
    require_valid_index(index, size);

    if (index >= 0)
    {
        return static_cast<size_type>(index);
    }

    return size - detail::negative_magnitude(index);
}

} // namespace stratax::validation
