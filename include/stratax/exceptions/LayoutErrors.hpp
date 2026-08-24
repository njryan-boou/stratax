#pragma once

#include <cstddef>
#include <optional>
#include <source_location>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <stratax/exceptions/StrataxError.hpp>

namespace Exceptions {

class ShapeError : public StrataxError
{
public:
    using shape_type = std::vector<std::size_t>;
    explicit ShapeError(std::string message, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Shape, location) {}
    explicit ShapeError(shape_type actual, std::source_location location = std::source_location::current())
        : StrataxError(make_message(actual, std::nullopt), ErrorCode::Shape, location), actual_(std::move(actual)) {}
    ShapeError(shape_type actual, shape_type expected, std::source_location location = std::source_location::current())
        : StrataxError(make_message(actual, expected), ErrorCode::Shape, location), actual_(std::move(actual)), expected_(std::move(expected)) {}
    ShapeError(std::string message, shape_type actual, std::optional<shape_type> expected = std::nullopt, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Shape, location), actual_(std::move(actual)), expected_(std::move(expected)) {}

    [[nodiscard]] bool has_shape_metadata() const noexcept { return actual_.has_value(); }
    [[nodiscard]] const std::optional<shape_type>& actual_shape() const noexcept { return actual_; }
    [[nodiscard]] const std::optional<shape_type>& expected_shape() const noexcept { return expected_; }

    static ShapeError vector_conversion(shape_type shape, std::source_location location = std::source_location::current())
    {
        const std::string message = "Cannot convert array with shape " + format_shape(shape) + " to Vector: after ignoring zero and singleton dimensions, exactly one non-singleton dimension is required.";
        return {message, std::move(shape), std::nullopt, location};
    }
    static ShapeError matrix_conversion(shape_type shape, std::source_location location = std::source_location::current())
    {
        const std::string message = "Cannot convert array with shape " + format_shape(shape) + " to Matrix: after ignoring zero and singleton dimensions, exactly two non-singleton dimensions are required.";
        return {message, std::move(shape), std::nullopt, location};
    }
    static ShapeError reshape_size(shape_type shape, std::size_t source_size, std::size_t target_size, std::source_location location = std::source_location::current())
    {
        const std::string message = "Cannot reshape array containing " + std::to_string(source_size) + " elements to shape " + format_shape(shape) + ", which requires " + std::to_string(target_size) + " elements; reshape must preserve the element count.";
        return {message, std::move(shape), std::nullopt, location};
    }
    static ShapeError buffer_size(shape_type shape, std::size_t buffer_size, std::size_t required_size, std::source_location location = std::source_location::current())
    {
        const std::string message = "Cannot construct an array with shape " + format_shape(shape) + ": the buffer contains " + std::to_string(buffer_size) + " elements, but the shape requires exactly " + std::to_string(required_size) + ".";
        return {message, std::move(shape), std::nullopt, location};
    }
    static ShapeError ragged_matrix(std::source_location location = std::source_location::current()) { return ShapeError("Cannot construct a rectangular Matrix from ragged rows: every row must contain exactly the same number of elements as the first row.", location); }
    static ShapeError ragged_initializer(std::source_location location = std::source_location::current()) { return ShapeError("Cannot construct a rectangular Matrix from this initializer list: every row must contain exactly the same number of columns as the first row.", location); }
    static ShapeError vector_rank(shape_type shape, std::source_location location = std::source_location::current())
    {
        const std::string message = "Vector requires a rank-1 shape, but shape " + format_shape(shape) + " has rank " + std::to_string(shape.size()) + ".";
        return {message, std::move(shape), std::nullopt, location};
    }
    static ShapeError matrix_rank(shape_type shape, std::source_location location = std::source_location::current())
    {
        const std::string message = "Matrix requires a rank-2 shape, but shape " + format_shape(shape) + " has rank " + std::to_string(shape.size()) + ".";
        return {message, std::move(shape), std::nullopt, location};
    }
    static ShapeError negative_shape_dimension(std::source_location location = std::source_location::current()) { return ShapeError("Cannot construct a Shape with a negative dimension; every dimension must be zero or greater.", location); }
    static ShapeError negative_tensor_dimension(std::source_location location = std::source_location::current()) { return ShapeError("Cannot construct a Tensor with a negative shape dimension; every dimension must be zero or greater.", location); }

private:
    static std::string format_shape(const shape_type& shape)
    {
        std::ostringstream stream;
        stream << '(';
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            if (i != 0) stream << ", ";
            stream << shape[i];
        }
        return stream << ')', stream.str();
    }
    static std::string make_message(const shape_type& actual, const std::optional<shape_type>& expected)
    {
        if (expected) return "Shape " + format_shape(actual) + " does not match expected shape " + format_shape(*expected) + '.';
        return "Invalid shape " + format_shape(actual) + '.';
    }
    std::optional<shape_type> actual_;
    std::optional<shape_type> expected_;
};

class DimensionError : public StrataxError
{
public:
    using difference_type = std::ptrdiff_t;
    explicit DimensionError(std::string message, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Dimension, location) {}
    DimensionError(difference_type actual, difference_type expected, std::source_location location = std::source_location::current())
        : StrataxError("Dimension " + std::to_string(actual) + " does not match expected dimension " + std::to_string(expected) + '.', ErrorCode::Dimension, location), actual_(actual), expected_(expected) {}
    DimensionError(std::string message, difference_type actual, std::optional<difference_type> expected = std::nullopt, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Dimension, location), actual_(actual), expected_(expected) {}

    [[nodiscard]] bool has_dimension_metadata() const noexcept { return actual_.has_value(); }
    [[nodiscard]] const std::optional<difference_type>& actual_dimension() const noexcept { return actual_; }
    [[nodiscard]] const std::optional<difference_type>& expected_dimension() const noexcept { return expected_; }

    static DimensionError negative_vector_size(difference_type value, std::source_location location = std::source_location::current()) { return {"Vector size " + std::to_string(value) + " is invalid; vector sizes must be nonnegative.", value, 0, location}; }
    static DimensionError negative_matrix_rows(difference_type value, std::source_location location = std::source_location::current()) { return {"Matrix row count " + std::to_string(value) + " is invalid; row counts must be nonnegative.", value, 0, location}; }
    static DimensionError negative_matrix_columns(difference_type value, std::source_location location = std::source_location::current()) { return {"Matrix column count " + std::to_string(value) + " is invalid; column counts must be nonnegative.", value, 0, location}; }
    static DimensionError slice_rank(difference_type actual, difference_type expected, std::source_location location = std::source_location::current()) { return {"Received " + std::to_string(actual) + " slice components for a rank-" + std::to_string(expected) + " tensor; provide exactly one slice per tensor dimension.", actual, expected, location}; }
    static DimensionError shape_elements_overflow(std::source_location location = std::source_location::current()) { return DimensionError("Cannot compute the shape's element count because multiplying its dimensions exceeds the maximum representable size; reduce one or more dimensions.", location); }
    static DimensionError stride_overflow(std::source_location location = std::source_location::current()) { return DimensionError("Cannot compute row-major strides because multiplying trailing dimensions exceeds the maximum representable size; reduce one or more dimensions.", location); }
    static DimensionError tensor_offset_overflow(std::source_location location = std::source_location::current()) { return DimensionError("Cannot create the tensor view because its computed storage offset exceeds the maximum representable index.", location); }
    static DimensionError tensor_stride_overflow(std::source_location location = std::source_location::current()) { return DimensionError("Cannot create the tensor view because a slice step multiplied by its source stride exceeds the maximum representable stride.", location); }

private:
    std::optional<difference_type> actual_;
    std::optional<difference_type> expected_;
};

class BroadcastError : public StrataxError
{
public:
    using shape_type = std::vector<std::size_t>;
    explicit BroadcastError(std::string message, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Broadcast, location) {}
    BroadcastError(shape_type left, shape_type right, std::source_location location = std::source_location::current())
        : StrataxError("Shapes are not broadcastable.", ErrorCode::Broadcast, location), left_(std::move(left)), right_(std::move(right)) {}
    BroadcastError(std::string message, shape_type left, shape_type right, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Broadcast, location), left_(std::move(left)), right_(std::move(right)) {}

    [[nodiscard]] bool has_shape_metadata() const noexcept { return left_.has_value(); }
    [[nodiscard]] const std::optional<shape_type>& left_shape() const noexcept { return left_; }
    [[nodiscard]] const std::optional<shape_type>& right_shape() const noexcept { return right_; }

    static BroadcastError incompatible(shape_type left, shape_type right, std::source_location location = std::source_location::current())
    {
        const std::string message = "Cannot broadcast shapes " + format_shape(left) + " and " + format_shape(right);
        return {message, std::move(left), std::move(right), location};
    }
    static BroadcastError compound_arithmetic(shape_type result, shape_type left, std::source_location location = std::source_location::current()) { const auto message = "Compound arithmetic would produce shape " + format_shape(result) + ", but the left operand has shape " + format_shape(left) + "; an in-place operation cannot change the left operand's shape."; return {message, std::move(result), std::move(left), location}; }
    static BroadcastError compound_bitwise(shape_type result, shape_type left, std::source_location location = std::source_location::current()) { const auto message = "Compound bitwise arithmetic would produce shape " + format_shape(result) + ", but the left operand has shape " + format_shape(left) + "; an in-place operation cannot change the left operand's shape."; return {message, std::move(result), std::move(left), location}; }
    static BroadcastError compound_shift(shape_type result, shape_type left, std::source_location location = std::source_location::current()) { const auto message = "Compound shifting would produce shape " + format_shape(result) + ", but the left operand has shape " + format_shape(left) + "; an in-place operation cannot change the left operand's shape."; return {message, std::move(result), std::move(left), location}; }

private:
    static std::string format_shape(const shape_type& shape)
    {
        std::ostringstream stream;
        stream << '(';
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            if (i != 0) stream << ", ";
            stream << shape[i];
        }
        return stream << ')', stream.str();
    }

    std::optional<shape_type> left_;
    std::optional<shape_type> right_;
};

} // namespace Exceptions
