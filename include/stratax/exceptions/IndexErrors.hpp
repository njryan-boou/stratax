#pragma once

#include <cstddef>
#include <optional>
#include <source_location>
#include <sstream>
#include <string>
#include <utility>

#include <stratax/exceptions/StrataxError.hpp>

namespace Exceptions {

class IndexError : public StrataxError
{
public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    enum class Context { Array, Tensor, VectorSlice, MatrixRowSlice, MatrixColumnSlice, TensorSlice };

    explicit IndexError(std::string message, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Index, location) {}
    IndexError(difference_type index, size_type size, std::source_location location = std::source_location::current())
        : StrataxError(make_message(index, size), ErrorCode::Index, location), index_(index), size_(size) {}
    IndexError(std::string message, difference_type index, size_type size, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Index, location), index_(index), size_(size) {}

    [[nodiscard]] bool has_index_metadata() const noexcept { return index_.has_value(); }
    [[nodiscard]] const std::optional<difference_type>& index() const noexcept { return index_; }
    [[nodiscard]] const std::optional<size_type>& size() const noexcept { return size_; }

    static IndexError slice_step_zero(std::source_location location = std::source_location::current()) { return IndexError("Slice step cannot be zero because it would never advance to another element; use a positive or negative nonzero step.", location); }
    static IndexError negative_step_view(std::source_location location = std::source_location::current()) { return IndexError("Cannot create a view from a slice with a negative step: reverse-strided views are not supported; use a positive step or copy the reversed values.", location); }
    static IndexError tensor_slice_rank(size_type actual, size_type expected, std::source_location location = std::source_location::current()) { return rank_error("Tensor slice", actual, expected, location); }
    static IndexError tensor_index_rank(size_type actual, size_type expected, std::source_location location = std::source_location::current()) { return rank_error("Tensor index", actual, expected, location); }
    static IndexError matrix_index_rank(size_type actual, std::source_location location = std::source_location::current()) { return rank_error("Matrix index", actual, 2, location); }
    static IndexError matrix_tuple_index(size_type actual, std::source_location location = std::source_location::current()) { return rank_error("Matrix assignment index", actual, 2, location); }
    static IndexError view_index_rank(size_type actual, size_type expected, std::source_location location = std::source_location::current()) { return rank_error("ArrayView index", actual, expected, location); }
    static IndexError array_view_index_rank(size_type actual, size_type expected, std::source_location location = std::source_location::current()) { return rank_error("ArrayView checked index", actual, expected, location); }
    static IndexError multi_index_rank(size_type actual, size_type expected, Context context = Context::Array, std::source_location location = std::source_location::current())
    {
        return rank_error(context == Context::Tensor ? "Tensor multi-index" : "Array multi-index", actual, expected, location);
    }
    static IndexError multi_index_component(difference_type index, size_type size, Context context = Context::Array, std::source_location location = std::source_location::current())
    {
        const std::string prefix = context == Context::Tensor ? "Tensor multi-index component is invalid: " : "Multi-index component is invalid: ";
        return {prefix + make_message(index, size), index, size, location};
    }
    static IndexError slice_extent(Context context, size_type extent, std::source_location location = std::source_location::current())
    {
        const std::string suffix = " extent " + std::to_string(extent) + " exceeds the largest value representable by the signed index type; use a smaller dimension.";
        switch (context)
        {
        case Context::VectorSlice: return IndexError("Vector slice" + suffix, location);
        case Context::MatrixRowSlice: return IndexError("Matrix row slice" + suffix, location);
        case Context::MatrixColumnSlice: return IndexError("Matrix column slice" + suffix, location);
        default: return IndexError("Tensor slice" + suffix, location);
        }
    }
    static IndexError maximum_empty(std::source_location location = std::source_location::current()) { return IndexError("Cannot compute a maximum for an empty array because there is no element to select; provide at least one element.", location); }
    static IndexError minimum_empty(std::source_location location = std::source_location::current()) { return IndexError("Cannot compute a minimum for an empty array because there is no element to select; provide at least one element.", location); }
    static IndexError argmax_empty(std::source_location location = std::source_location::current()) { return IndexError("Cannot compute argmax for an empty array because no valid element index exists; provide at least one element.", location); }
    static IndexError argmin_empty(std::source_location location = std::source_location::current()) { return IndexError("Cannot compute argmin for an empty array because no valid element index exists; provide at least one element.", location); }

private:
    static IndexError rank_error(const std::string& subject, size_type actual, size_type expected, std::source_location location)
    {
        return {subject + " has " + std::to_string(actual) + " components, but the target has rank " + std::to_string(expected) + "; provide exactly one index or slice component per dimension.", static_cast<difference_type>(actual), expected, location};
    }
    static std::string make_message(difference_type index, size_type size)
    {
        std::ostringstream stream;
        stream << "Index " << index << " is out of bounds for size " << size << '.';
        if (size == 0) stream << " The array is empty.";
        else stream << " Valid indices range from -" << size << " through " << (size - 1) << '.';
        return stream.str();
    }

    std::optional<difference_type> index_;
    std::optional<size_type> size_;
};

class AxisError : public StrataxError
{
public:
    using difference_type = std::ptrdiff_t;
    explicit AxisError(std::string message, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Axis, location) {}
    AxisError(difference_type axis, std::size_t rank, std::source_location location = std::source_location::current())
        : StrataxError("Axis " + std::to_string(axis) + " is out of range for rank " + std::to_string(rank) + '.', ErrorCode::Axis, location), axis_(axis), rank_(rank) {}
    AxisError(std::string message, difference_type axis, std::size_t rank, std::source_location location = std::source_location::current())
        : StrataxError(std::move(message), ErrorCode::Axis, location), axis_(axis), rank_(rank) {}

    [[nodiscard]] bool has_axis_metadata() const noexcept { return axis_.has_value(); }
    [[nodiscard]] const std::optional<difference_type>& axis() const noexcept { return axis_; }
    [[nodiscard]] const std::optional<std::size_t>& rank() const noexcept { return rank_; }
    static AxisError out_of_range(difference_type axis, std::size_t rank, std::source_location location = std::source_location::current())
    {
        std::string message = "Axis " + std::to_string(axis) + " is out of range for an array of rank " + std::to_string(rank) + ".";
        if (rank == 0) message += " A rank-0 array has no valid axes.";
        else message += " Valid axes range from -" + std::to_string(rank) + " through " + std::to_string(rank - 1) + ".";
        return {std::move(message), axis, rank, location};
    }

private:
    std::optional<difference_type> axis_;
    std::optional<std::size_t> rank_;
};

} // namespace Exceptions
