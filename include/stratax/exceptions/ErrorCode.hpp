#pragma once

#include <string_view>

namespace Exceptions {

enum class ErrorCode
{
    Stratax,
    Shape,
    Dimension,
    Index,
    Type,
    Broadcast,
    ZeroDivision,
    Axis,
    Overflow,
    Value
};

[[nodiscard]] constexpr std::string_view error_code_name(
    ErrorCode code) noexcept
{
    switch (code)
    {
    case ErrorCode::Stratax: return "stratax";
    case ErrorCode::Shape: return "shape";
    case ErrorCode::Dimension: return "dimension";
    case ErrorCode::Index: return "index";
    case ErrorCode::Type: return "type";
    case ErrorCode::Broadcast: return "broadcast";
    case ErrorCode::ZeroDivision: return "zero_division";
    case ErrorCode::Axis: return "axis";
    case ErrorCode::Overflow: return "overflow";
    case ErrorCode::Value: return "value";
    }

    return "unknown";
}

} // namespace Exceptions
