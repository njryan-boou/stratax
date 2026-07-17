#pragma once

#include <cstddef>

#include <stratax/core/Exceptions.hpp>

namespace stratax::core::validation {

/**
 * @brief Converts a signed shape dimension to `std::size_t` after rejecting negatives.
 *
 * @param value Signed dimension to validate.
 * @param message Error message used when `value` is negative.
 *
 * @return `value` converted to `std::size_t`.
 *
 * @throws Exceptions::ShapeError If `value` is negative.
 */
inline std::size_t nonnegative_shape_dimension(std::ptrdiff_t value, const char* message)
{
    if (value < 0)
    {
        throw Exceptions::ShapeError(message);
    }

    return static_cast<std::size_t>(value);
}

/**
 * @brief Converts a signed shape dimension to `std::size_t` after requiring it to be positive.
 *
 * @param value Signed dimension to validate.
 * @param message Error message used when `value <= 0`.
 *
 * @return `value` converted to `std::size_t`.
 *
 * @throws Exceptions::ShapeError If `value <= 0`.
 */
inline std::size_t positive_shape_dimension(std::ptrdiff_t value, const char* message)
{
    if (value <= 0)
    {
        throw Exceptions::ShapeError(message);
    }

    return static_cast<std::size_t>(value);
}

/**
 * @brief Requires an unsigned shape dimension to be positive.
 *
 * @param value Dimension to validate.
 * @param message Error message used when `value == 0`.
 *
 * @throws Exceptions::ShapeError If `value == 0`.
 */
inline void require_positive_shape_dimension(std::size_t value, const char* message)
{
    if (value == 0)
    {
        throw Exceptions::ShapeError(message);
    }
}

/**
 * @brief Returns whether two array-like objects have identical size and shape.
 *
 * @param lhs Left object.
 * @param rhs Right object.
 *
 * @return `true` if size and shape match.
 */
template<typename Lhs, typename Rhs>
[[nodiscard]] bool same_shape(const Lhs& lhs, const Rhs& rhs)
{
    return lhs.size() == rhs.size() && lhs.shape() == rhs.shape();
}

/**
 * @brief Requires two array-like objects to have identical size and shape.
 *
 * @param lhs Left object.
 * @param rhs Right object.
 * @param message Error message used on mismatch.
 *
 * @throws Exceptions::ShapeError If size or shape differ.
 */
template<typename Lhs, typename Rhs>
void require_same_shape(const Lhs& lhs, const Rhs& rhs, const char* message)
{
    if (!same_shape(lhs, rhs))
    {
        throw Exceptions::ShapeError(message);
    }
}

/**
 * @brief Requires two sizes to be equal.
 *
 * @param lhs Left size.
 * @param rhs Right size.
 * @param message Error message used on mismatch.
 *
 * @throws Exceptions::ShapeError If the sizes differ.
 */
inline void require_equal_size(std::size_t lhs, std::size_t rhs, const char* message)
{
    if (lhs != rhs)
    {
        throw Exceptions::ShapeError(message);
    }
}

}
