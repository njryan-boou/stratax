#pragma once

#include <string>
#include <utility>

#include <stratax/exceptions/TypeErrors.hpp>

namespace stratax::validation {

inline void require_type(
    bool matches,
    std::string actual,
    std::string expected)
{
    if (matches) return;

    throw Exceptions::TypeError(
        std::move(actual),
        std::move(expected));
}

} // namespace stratax::validation
