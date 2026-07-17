#pragma once

#include <vector>

#include "Buffer.hpp"
#include "../Concepts.hpp"
#include "../Exceptions.hpp"
#include "../validation/Validation.hpp"

#include <ostream>

namespace stratax::core {

/**
 * @brief Stores a list of dimension lengths for an array shape.
 *
 * Shape tracks rank, dimension sizes, and derived element counts for
 * Stratax containers. It provides lightweight metadata only and does
 * not own any element storage.
 */
class Shape
{
private:
    Buffer<std::size_t> dims_;

    void validate_dimensions() const
    {
        for (std::size_t dim : dims_)
        {
            validation::nonnegative_shape_dimension(
                dim,
                "Shape dimensions cannot be negative.");
        }
    }

public:
    struct allow_zero_t {};
    static constexpr allow_zero_t allow_zero{};

    /**
     * @brief Creates an empty shape.
     *
     * An empty shape has rank 0 and stores no axis lengths.
     */
    Shape() noexcept = default;

    /**
     * @brief Creates a shape from non-negative integer dimension lengths.
     *
     * The dimensions are stored in order, from outermost axis to
     * innermost axis. Boolean and character-like types are rejected by
     * the `Integral` concept.
     *
     * @param dims Dimension lengths.
     */
    template<Integral... Dims>
    requires (sizeof...(Dims) > 0)
    Shape(Dims... dims)
        : dims_{static_cast<std::size_t>(dims)...}
    {
        validate_dimensions();
    }

    /**
     * @brief Creates a shape from dimension lengths using the explicit zero-dimension tag.
     *
     * This overload documents zero-preserving construction for call sites that
     * want to make that intent visible.
     *
     * @param list Dimension lengths.
     * @param allow_zero Tag documenting that zero-valued dimensions are intentional.
     */
    Shape(std::initializer_list<std::size_t> list, allow_zero_t allow_zero) : dims_(list)
    {
        (void)allow_zero;
    }

    /**
     * @brief Creates a shape by copying dimension lengths from a buffer.
     *
     * @param dims Source buffer containing one dimension length per entry.
     */
    Shape(const Buffer<std::size_t>& dims)
    : dims_(dims)
    {
        validate_dimensions();
    }

    /**
     * @brief Creates a shape by copying dimension lengths using the explicit zero-dimension tag.
     *
     * @param dims Source buffer containing one dimension length per entry.
     * @param allow_zero Tag documenting that zero-valued dimensions are intentional.
     */
    Shape(const Buffer<std::size_t>& dims, allow_zero_t allow_zero)
    : dims_(dims)
    {
        (void)allow_zero;
    }

    /**
     * @brief Creates a shape by taking ownership of dimension lengths from a buffer.
     *
     * @param dims Source buffer containing one dimension length per entry.
     */
    Shape(Buffer<std::size_t>&& dims)
    : dims_(std::move(dims))
    {
        validate_dimensions();
    }

    /**
     * @brief Creates a shape by copying dimension lengths from a standard vector.
     *
     * @param dims Source vector containing one dimension length per entry.
     */
    Shape(const std::vector<std::size_t>& dims)
        : dims_(dims.size())
    {
        for (std::size_t i = 0; i < dims.size(); ++i)
        {
            dims_[i] = dims[i];
        }
    }

    /**
     * @brief Destroys the shape.
     */
    ~Shape() = default;

    /**
     * @brief Returns the total number of elements described by the shape.
     *
     * The result is the product of all stored dimensions. An empty shape
     * reports zero elements.
     *
     * @return Total number of elements implied by the shape.
     *
     * @throws Exceptions::DimensionError If the product of dimensions would overflow `std::size_t`.
     */
    [[nodiscard]]
    std::size_t elements() const
    {
        if (empty())
        {
            return 0;
        }
        std::size_t prod = 1;
        for (std::size_t dim : dims_)
        {
            prod = validation::checked_multiply(prod, dim, "Shape elements overflow");
        }
        return prod;
    }

    /**
     * @brief Returns the number of stored dimensions.
     *
     * @return Shape rank.
     */
    [[nodiscard]] std::size_t rank() const
    {
        return dims_.size();
    }

    /**
     * @brief Returns the length of a specific dimension.
     *
     * @param index Zero-based dimension index.
     *
     * @return Dimension length at the requested index.
     *
     * @throws Exceptions::IndexError If `index` is greater than or equal to the shape rank.
     */
    const std::size_t& operator()(std::size_t index) const
    {
        validation::require_index(index, rank(), "Shape dimension index out of bounds");
        return dims_[index];
    }

    /**
     * @brief Returns the length of a specific dimension using signed indexing.
     *
     * Negative indices address dimensions from the end of the shape.
     *
     * @param index Signed dimension index.
     *
     * @return Dimension length at the requested index.
     *
     * @throws Exceptions::IndexError If `index` is out of bounds.
     */
    const std::size_t& operator[](std::ptrdiff_t index) const
    {
        return dims_[validation::normalize_index(index, rank(), "Shape dimension index out of bounds")];
    }

    /**
     * @brief Returns whether the shape has no dimensions.
     *
     * @return `true` when the shape rank is zero.
     */
    [[nodiscard]] bool empty() const noexcept
    {
        return dims_.empty();
    }

    /**
     * @brief Compares two shapes for exact rank and dimension equality.
     *
     * @param other Shape to compare against.
     *
     * @return `true` when both shapes have the same rank and dimension values.
     */
    [[nodiscard]] bool operator==(const Shape& other) const noexcept
    {
        if (rank() != other.rank())
        {
            return false;
        }
        for (std::size_t i = 0; i < rank(); ++i)
        {
            if (dims_[i] != other.dims_[i])
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Returns whether two shapes differ in rank or dimension values.
     *
     * @param other Shape to compare against.
     *
     * @return `true` when the shapes are not equal.
     */
    [[nodiscard]] bool operator!=(const Shape& other) const noexcept
    {
        return !(*this == other);
    }

    /**
     * @brief Returns an iterator to the first stored dimension.
     *
     * @return Iterator to the beginning of the dimension storage.
     */
    auto begin() noexcept
    {
        return dims_.begin();
    }

    /**
     * @brief Returns an iterator one past the last stored dimension.
     *
     * @return Iterator to the end of the dimension storage.
     */
    auto end() noexcept
    {
        return dims_.end();
    }

    /**
     * @brief Returns a const iterator to the first stored dimension.
     *
     * @return Const iterator to the beginning of the dimension storage.
     */
    auto begin() const noexcept
    {
        return dims_.begin();
    }

    /**
     * @brief Returns a const iterator one past the last stored dimension.
     *
     * @return Const iterator to the end of the dimension storage.
     */
    auto end() const noexcept
    {
        return dims_.end();
    }

    /**
     * @brief Swaps the stored dimensions with another shape.
     *
     * @param other Shape to exchange state with.
     */
    void swap(Shape& other) noexcept
    {
        dims_.swap(other.dims_);
    }

};

/**
 * @brief Writes a shape to a stream in tuple-like form.
 *
 * @param os Output stream.
 * @param shape Shape to print.
 *
 * @return The updated output stream.
 */
inline std::ostream& operator<<(std::ostream& os, const Shape& shape)
{
    os << "(";

    bool first = true;
    for (std::size_t dim : shape)
    {
        if (!first)
            os << ", ";

        os << dim;
        first = false;
    }

    if (shape.rank() == 1)
    {
        os << ",";
    }

    os << ")";

    return os;
}

}

