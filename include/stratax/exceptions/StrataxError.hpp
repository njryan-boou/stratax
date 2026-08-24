#pragma once

#include <cstdint>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <stratax/exceptions/ErrorCode.hpp>

namespace Exceptions {

class StrataxError : public std::runtime_error
{
public:
    explicit StrataxError(
        std::string message,
        std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Stratax, location)
    {}

    [[nodiscard]] ErrorCode code() const noexcept { return code_; }
    [[nodiscard]] std::string_view category() const noexcept
    {
        return error_code_name(code_);
    }
    [[nodiscard]] const std::source_location& location() const noexcept
    {
        return location_;
    }
    [[nodiscard]] std::string_view file_name() const noexcept
    {
        return location_.file_name();
    }
    [[nodiscard]] std::string_view function_name() const noexcept
    {
        return location_.function_name();
    }
    [[nodiscard]] std::uint_least32_t line() const noexcept
    {
        return location_.line();
    }
    [[nodiscard]] std::uint_least32_t column() const noexcept
    {
        return location_.column();
    }

    [[nodiscard]] std::string diagnostic() const
    {
        std::ostringstream stream;
        stream << category() << ": " << what();

        if (!file_name().empty())
        {
            stream << " [" << file_name() << ':' << line();
            if (column() != 0) stream << ':' << column();
            stream << ']';
        }

        return stream.str();
    }

protected:
    StrataxError(
        std::string message,
        ErrorCode code,
        std::source_location location)
        : std::runtime_error(std::move(message)),
          code_(code),
          location_(location)
    {}

private:
    ErrorCode code_;
    std::source_location location_;
};

} // namespace Exceptions
