#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include <stratax/exceptions/LayoutErrors.hpp>

namespace stratax::validation {

inline void require_nonnegative_dimension(
    std::ptrdiff_t value)
{
    if (value < 0)
    {
        throw Exceptions::DimensionError(value, 0);
    }
}

inline void require_dimension(
    std::size_t actual,
    std::size_t expected)
{
    if (actual == expected) return;

    throw Exceptions::DimensionError(
        static_cast<std::ptrdiff_t>(actual),
        static_cast<std::ptrdiff_t>(expected));
}

} // namespace stratax::validation
