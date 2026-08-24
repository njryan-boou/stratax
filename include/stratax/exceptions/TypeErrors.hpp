#pragma once

#include <optional>
#include <source_location>
#include <string>
#include <utility>

#include <stratax/exceptions/StrataxError.hpp>

namespace Exceptions {

class TypeError : public StrataxError
{
public:
    explicit TypeError(std::string message, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Type, location) {}
    TypeError(std::string actual, std::string expected, std::source_location location = std::source_location::current())
        : StrataxError("Type '" + actual + "' does not match expected type '" + expected + "'.", ErrorCode::Type, location), actual_(std::move(actual)), expected_(std::move(expected)) {}
    TypeError(std::string message, std::string actual, std::string expected, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Type, location), actual_(std::move(actual)), expected_(std::move(expected)) {}

    [[nodiscard]] bool has_type_metadata() const noexcept { return actual_.has_value(); }
    [[nodiscard]] const std::optional<std::string>& actual_type() const noexcept { return actual_; }
    [[nodiscard]] const std::optional<std::string>& expected_type() const noexcept { return expected_; }

    static TypeError expected_integer(std::string actual = "Python object", std::source_location location = std::source_location::current())
    {
        const std::string message = "Expected an integer value, but received " + actual + "; boolean and non-integral values are not accepted for indices or dimensions.";
        return {message, std::move(actual), "integer", location};
    }
    static TypeError expected_number(std::string actual = "Python object", std::source_location location = std::source_location::current())
    {
        const std::string message = "Expected a numeric value, but received " + actual + "; provide an integer, floating-point, or supported complex value.";
        return {message, std::move(actual), "number", location};
    }
    static TypeError vector_constructor(std::string actual = "Python object", std::source_location location = std::source_location::current()) { return {"Vector constructor expects a Vector, size, or iterable of numbers.", std::move(actual), "Vector, size, or iterable of numbers", location}; }
    static TypeError matrix_rows(std::string actual = "Python object", std::source_location location = std::source_location::current()) { return {"Matrix rows must be iterables of numbers.", std::move(actual), "iterable of numbers", location}; }
    static TypeError matrix_constructor(std::string actual = "Python object", std::source_location location = std::source_location::current()) { return {"Matrix constructor expects an iterable of row values.", std::move(actual), "iterable of row values", location}; }
    static TypeError tensor_shape(std::string actual = "Python object", std::source_location location = std::source_location::current()) { return {"Tensor shape must be a Shape or iterable of dimensions.", std::move(actual), "Shape or iterable of dimensions", location}; }

private:
    std::optional<std::string> actual_;
    std::optional<std::string> expected_;
};

class ValueError : public StrataxError
{
public:
    explicit ValueError(std::string message, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Value, location) {}
    static ValueError shift_count(std::source_location location = std::source_location::current())
    {
        return ValueError("Shift count must be non-negative and less than the bit width of the shifted value.", location);
    }
};

} // namespace Exceptions
