#pragma once

#include <cstddef>
#include <optional>
#include <source_location>
#include <string>
#include <utility>

#include <stratax/exceptions/StrataxError.hpp>

namespace Exceptions {

class ZeroDivisionError : public StrataxError
{
public:
    explicit ZeroDivisionError(std::string message, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::ZeroDivision, location) {}
    explicit ZeroDivisionError(std::size_t index, std::source_location location = std::source_location::current())
        : StrataxError("Division by zero at flat index " + std::to_string(index) + '.', ErrorCode::ZeroDivision, location), index_(index) {}
    ZeroDivisionError(std::string message, std::size_t index, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::ZeroDivision, location), index_(index) {}

    [[nodiscard]] bool has_index_metadata() const noexcept { return index_.has_value(); }
    [[nodiscard]] const std::optional<std::size_t>& index() const noexcept { return index_; }

    static ZeroDivisionError array_element(std::source_location location = std::source_location::current()) { return ZeroDivisionError("Element-wise array division encountered a zero in the broadcasted divisor; every divisor element must be nonzero.", location); }
    static ZeroDivisionError array_scalar(std::source_location location = std::source_location::current()) { return ZeroDivisionError("Cannot divide an array by scalar zero; provide a nonzero scalar divisor.", location); }
    static ZeroDivisionError scalar_element(std::source_location location = std::source_location::current()) { return ZeroDivisionError("Scalar-by-array division encountered a zero divisor element; every array element used as a divisor must be nonzero.", location); }
    static ZeroDivisionError compound_element(std::size_t index, std::source_location location = std::source_location::current()) { return {"Compound array division encountered a zero divisor at flat index " + std::to_string(index) + "; every divisor element must be nonzero.", index, location}; }
    static ZeroDivisionError compound_scalar(std::source_location location = std::source_location::current()) { return ZeroDivisionError("Cannot perform compound array division by scalar zero; provide a nonzero scalar divisor.", location); }
    static ZeroDivisionError mean_empty(std::source_location location = std::source_location::current()) { return ZeroDivisionError("Cannot compute the mean of an empty array because the element count is zero; provide at least one element.", location); }
    static ZeroDivisionError variance_empty(std::source_location location = std::source_location::current()) { return ZeroDivisionError("Cannot compute variance of an empty array because the element count is zero; provide at least one element.", location); }

private:
    std::optional<std::size_t> index_;
};

class OverflowError : public StrataxError
{
public:
    explicit OverflowError(std::string message, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Overflow, location) {}
    OverflowError(std::string operation, std::size_t left, std::size_t right, std::source_location location = std::source_location::current())
        : StrataxError(operation + " overflowed for operands " + std::to_string(left) + " and " + std::to_string(right) + '.', ErrorCode::Overflow, location), operation_(std::move(operation)), left_(left), right_(right) {}
    OverflowError(std::string message, std::string operation, std::size_t left, std::size_t right, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Overflow, location), operation_(std::move(operation)), left_(left), right_(right) {}

    [[nodiscard]] bool has_operand_metadata() const noexcept { return left_.has_value(); }
    [[nodiscard]] const std::optional<std::string>& operation() const noexcept { return operation_; }
    [[nodiscard]] const std::optional<std::size_t>& left_operand() const noexcept { return left_; }
    [[nodiscard]] const std::optional<std::size_t>& right_operand() const noexcept { return right_; }

    static OverflowError integer(std::source_location location = std::source_location::current()) { return OverflowError("Integer arithmetic produced a value outside the representable range of the result type.", location); }
    static OverflowError floating(std::source_location location = std::source_location::current()) { return OverflowError("Floating-point arithmetic produced a non-finite value outside the accepted result range.", location); }
    static OverflowError shape_elements(std::source_location location = std::source_location::current()) { return OverflowError("Shape element count exceeds the maximum representable size.", location); }
    static OverflowError python_slice_extent(std::source_location location = std::source_location::current()) { return OverflowError("Container is too large to slice with Python indices.", location); }
    static OverflowError matrix_size(std::size_t rows, std::size_t columns, std::source_location location = std::source_location::current()) { return {"Cannot compute matrix element count: " + std::to_string(rows) + " rows multiplied by " + std::to_string(columns) + " columns exceeds the maximum representable size.", "multiplication", rows, columns, location}; }
    static OverflowError matrix_storage(std::size_t elements, std::size_t element_size, std::source_location location = std::source_location::current()) { return {"Cannot allocate matrix storage: " + std::to_string(elements) + " elements multiplied by " + std::to_string(element_size) + " bytes per element exceeds the maximum representable byte count.", "multiplication", elements, element_size, location}; }
    static OverflowError tensor_shape(std::size_t elements, std::size_t dimension, std::source_location location = std::source_location::current()) { return {"Cannot compute tensor element count: the partial count " + std::to_string(elements) + " multiplied by dimension " + std::to_string(dimension) + " exceeds the maximum representable size.", "multiplication", elements, dimension, location}; }
    static OverflowError tensor_storage(std::size_t elements, std::size_t element_size, std::source_location location = std::source_location::current()) { return {"Cannot allocate tensor storage: " + std::to_string(elements) + " elements multiplied by " + std::to_string(element_size) + " bytes per element exceeds the maximum representable byte count.", "multiplication", elements, element_size, location}; }

private:
    std::optional<std::string> operation_;
    std::optional<std::size_t> left_;
    std::optional<std::size_t> right_;
};

} // namespace Exceptions
